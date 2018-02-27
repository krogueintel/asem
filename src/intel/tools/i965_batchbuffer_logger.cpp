/*
 * Copyright © 2017 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <mutex>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <list>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <typeinfo>
#include <memory>
#include <functional>

#include <stdarg.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <assert.h>
#include <dlfcn.h>
#include <stdint.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <pthread.h>

#include "i965_batchbuffer_logger_instructions.h"
#include "drm-uapi/i915_drm.h"
#include "common/gen_decoder.h"
#include "gen_disasm.h"
#include "util/mesa-sha1.h"
#include "util/macros.h"

#include "tools/i965_batchbuffer_logger_app.h"
#include "tools/i965_batchbuffer_logger_output.h"
#include "tools/i965_batchbuffer_logger.h"

/* Basic overview of implementation:
 *  - BatchbufferLogger is a singleton to allow for calls into it
 *    without needing the object itself
 *
 *  - Using the driver provided function pointer, it "knows" what
 *    is considered the active batchbuffer
 *      * NOTE: before being initialized by a driver, the function
 *        pointer specifying the active batchbuffer returns a value
 *        indicating that there is no active batchbuffer
 *
 *  - BatchbufferLogger has a map keyed by file descriptor of
 *    GEMBufferTracker objects. A GEMBufferTracker has within it
 *      * a map keyed by GEM BO handle of GEMBufferObjects
 *      * a map keyed by GEM BO handle of BatchbufferLog
 *      * a dummy BatchbufferLog object
 *
 *  - A BatchbufferLog object is essentially a log of what
 *    API calls are made when in a batchbuffer
 *
 *  - A BatchbufferLog object is added to a GEMBufferTracker
 *    whenever a GEM BO handle not seen before is emitted by
 *    the function pointer provided by the driver that gives the
 *    active batchbuffer.
 *
 * - Whenever an entry is added to a BatchbufferLog object A,
 *   if there are any entries in the dummy BatchbufferLog
 *   those entries are moved to the BatchbufferLog A in a way
 *   so that when BatchbufferLog is printed to file, the entries
 *   from dummy come first.
 */

namespace {

template<typename T>
T
read_from_environment(const char *env, T default_value)
{
  const char *tmp;
  T return_value(default_value);

  tmp = std::getenv(env);
  if (tmp != nullptr) {
     std::string str(tmp);
     std::istringstream istr(tmp);
     istr >> return_value;
  }

  return return_value;
}

/* field extraction routines and helpers taken from
 * gen_decoder.c
 */
uint64_t
mask(int start, int end)
{
   uint64_t v;
   v = ~0ULL >> (63 - end + start);
   return v << start;
}

void
get_start_end_pos(int *start, int *end)
{
   if (*end - *start > 32) {
      int len = *end - *start;
      *start = *start % 32;
      *end = *start + len;
   } else {
      *start = *start % 32;
      *end = *end % 32;
   }
}

template<typename T>
T
field(uint64_t value, int start, int end)
{
   uint64_t v;
   get_start_end_pos(&start, &end);
   v = (value & mask(start, end)) >> (start);
   return static_cast<T>(v);
}

uint32_t
diff_helper(uint32_t from, uint32_t to)
{
   return std::max(to, from) - from;
}

class NonCopyable;
class GPUCommandCounterValue;
class PipeControlCounterValue;
class GPUCommandCounter;
class BatchbufferLoggerSession;
class BatchbufferLoggerOutput;
class MessageActionBase;
class MessageItem;
class MessageActionList;
class APIStartCallMarker;
class GEMBufferObject;
class GPUCommandFieldValue;
class GPUAddressQuery;
class GPUCommand;
class i965LatchState;
class i965Registers;
class i965HWContextData;
class GPUState;
class BatchRelocs;
class ShaderFileList;
class BatchbufferDecoder;
class BatchbufferLog;
class PerHWContext;
class GEMBufferTracker;
class ManagedGEMBuffer;
class BatchbufferLogger;
class ForceExistanceOfBatchbufferLogger;

class NonCopyable {
public:
   NonCopyable(void)
   {}

private:
   NonCopyable(const NonCopyable&) = delete;

   NonCopyable&
   operator=(const NonCopyable&) = delete;
};

class GPUCommandCounterValue {
public:
   GPUCommandCounterValue(void);

   /* Create GPUCommandCounterValue as a difference
    * between two GPUCommandCounterValue values;
    */
   GPUCommandCounterValue(const GPUCommandCounterValue &from,
                          const GPUCommandCounterValue &to);

   /* The instructiont behind q must match m_inst
    * OR m_inst must be nullptr
    */
   void
   increment(const GPUCommand &q);

   uint32_t
   accumulated_count(void) const
   {
      return m_accumulated_count;
   }

   uint32_t
   accumulated_length(void) const
   {
      return m_accumulated_length;
   }

   struct gen_group*
   inst(void) const
   {
      return m_inst;
   }

   bool
   non_zero(void) const
   {
      return m_accumulated_count != 0
         || m_accumulated_length != 0;
   }

   void
   reset(void)
   {
      m_accumulated_count = 0;
      m_accumulated_length = 0;
   }
private:
   struct gen_group *m_inst;
   uint32_t m_accumulated_count;
   uint32_t m_accumulated_length;
};

const char *PipeControlFields[] = {
   "Flush LLC",
   "Command Streamer Stall Enable",
   "Generic Media State Clear",
   "TLB Invalidate",
   "Depth Stall Enable",
   "Render Target Cache Flush Enable",
   "Instruction Cache Invalidate Enable",
   "Texture Cache Invalidation Enable",
   "Indirect State Pointers Disable",
   "Notify Enable",
   "Pipe Control Flush Enable",
   "DC Flush Enable",
   "VF Cache Invalidation Enable",
   "Constant Cache Invalidation Enable",
   "State Cache Invalidation Enable",
   "Stall At Pixel Scoreboard",
   "Depth Cache Flush Enable",
};

class PipeControlCounterValue {
public:
   enum {
      number_entries = ARRAY_SIZE(PipeControlFields)
   };

   PipeControlCounterValue(void);
   PipeControlCounterValue(const PipeControlCounterValue &from,
                           const PipeControlCounterValue &to);

   void
   increment(const GPUCommand &q);

   uint32_t
   accumulated_count(void) const
   {
      return m_base.accumulated_count();
   }

   uint32_t
   accumulated_length(void) const
   {
      return m_base.accumulated_length();
   }

   struct gen_group*
   inst(void) const
   {
      return m_base.inst();
   }

   bool
   counts_non_zero(void) const;

   bool
   non_zero(void) const
   {
      return m_base.non_zero()
         || counts_non_zero();
   }

   uint32_t
   operator[](unsigned int i) const
   {
      assert(i < number_entries);
      return m_counts[i];
   }

   void
   reset(void)
   {
      m_base.reset();
      std::fill(m_counts, m_counts + number_entries, 0);
   }

private:
   GPUCommandCounterValue m_base;
   uint32_t m_counts[number_entries];
};

/* GPUCommandCounter counts how many times each type of
 * GPU command is encountered and the total sum-length
 * of their encounters.
 */
class GPUCommandCounter {
public:
   typedef std::map<uint32_t, GPUCommandCounterValue> value_type;

   /* creates a GPUCommandCounter that is empty.
    */
   GPUCommandCounter():
      m_total_length(0),
      m_total_count(0),
      m_number_batchbuffers(0)
   {}

   /* creates a GPUCommandCounter that is the difference
    * between two GPUCommandCounter values; elements present
    * in the lhs that are not present in the rhs are ignored
    */
   GPUCommandCounter(const GPUCommandCounter &from,
                     const GPUCommandCounter &to);

   void
   increment(const GPUCommand &q);

   void
   add_batch(void)
   {
      ++m_number_batchbuffers;
   }

   bool
   empty(void) const
   {
      return m_values.empty()
         && m_total_length == 0
         && m_total_count == 0
         && m_number_batchbuffers == 0;
   }

   const value_type&
   values(void) const
   {
      return m_values;
   }

   const PipeControlCounterValue&
   pipe_value(void) const
   {
      return m_pipe_value;
   }

   uint32_t
   total_length(void) const
   {
      return m_total_length;
   }

   uint32_t
   total_count(void) const
   {
      return m_total_count;
   }

   uint32_t
   number_batchbuffers(void)
   {
      return m_number_batchbuffers;
   }

   void
   print(BatchbufferLoggerOutput &dst);

   void
   reset(void)
   {
      m_values.clear();
      m_pipe_value.reset();
      m_total_length = 0;
      m_total_count = 0;
      m_number_batchbuffers = 0;
   }

private:
   value_type m_values;
   PipeControlCounterValue m_pipe_value;
   uint32_t m_total_length;
   uint32_t m_total_count;
   uint32_t m_number_batchbuffers;
};

class BatchbufferLoggerSession:NonCopyable {
public:
   BatchbufferLoggerSession(const struct i965_batchbuffer_logger_session_params &params,
                            const GPUCommandCounter *counter,
                            bool add_begin_logging_block);

   ~BatchbufferLoggerSession(void);

   void
   add_element(enum i965_batchbuffer_logger_message_type_t tp,
               const void *name, uint32_t name_length,
               const void *value, uint32_t value_length);

   void
   add_element(enum i965_batchbuffer_logger_message_type_t tp,
               const std::vector<uint8_t> &name,
               const std::vector<uint8_t> &value)
   {
      const void *n, *v;
      n = (name.size() > 0) ? &name[0] : nullptr;
      v = (value.size() > 0) ? &value[0] : nullptr;
      add_element(tp, n, name.size(), v, value.size());
   }

   void
   pre_execbuffer2_ioctl(unsigned int id)
   {
      m_params.pre_execbuffer2_ioctl(m_params.client_data, id);
   }

   void
   post_execbuffer2_ioctl(unsigned int id)
   {
      m_params.post_execbuffer2_ioctl(m_params.client_data, id);
   }

   static
   void
   write_file(void *client_data,
              enum i965_batchbuffer_logger_message_type_t tp,
              const void *name, uint32_t name_length,
              const void *value, uint32_t value_length);

   static
   void
   flush_file(void *client_data, unsigned int id);

   static
   void
   close_file(void *client_data);

private:
   bool m_add_begin_logging_block;
   struct i965_batchbuffer_logger_session_params m_params;

   /* Counter value that is updated by BatchbufferLogger */
   const GPUCommandCounter *m_counter;

   /* counter value at time of open() */
   GPUCommandCounter m_at_time_of_open;
};

/* BatchbufferLoggerOutput purpose is to write the block
 * structure of a log to a collection of sessions
 */
class BatchbufferLoggerOutput:NonCopyable {
public:
   explicit
   BatchbufferLoggerOutput(void);

   ~BatchbufferLoggerOutput();

   void
   add_session(BatchbufferLoggerSession *p);

   bool
   remove_session(BatchbufferLoggerSession *p);

   void
   close_all_sessions(void);

   operator bool() const
   {
      return !m_sessions.empty();
   }

   void
   pre_execbuffer2_ioctl(unsigned int id);

   void
   post_execbuffer2_ioctl(unsigned int id);

   void
   begin_block(const char *txt);

   void
   begin_block_value(const char *txt, const char *fmt, ...);

   void
   vbegin_block_value(const char *txt, const char *fmt, va_list va);

   void
   end_block(void);

   void
   clear_block_stack(unsigned int desired_depth = 0);

   unsigned int
   current_block_level(void)
   {
      return m_block_stack.size();
   }

   void
   print_value(const char *name, const char *fmt, ...);

   void
   vprint_value(const char *name, const char *fmt, va_list va);

private:
   typedef std::vector<uint8_t> u8vector;
   typedef std::pair<u8vector, u8vector> block_stack_entry;

   void
   write_name_value(enum i965_batchbuffer_logger_message_type_t tp,
                    const char *name, const char *fmt,
                    va_list va);

   void
   add_element(enum i965_batchbuffer_logger_message_type_t tp,
               const void *name, uint32_t name_length,
               const void *value, uint32_t value_length);

   std::vector<block_stack_entry> m_block_stack;
   std::set<BatchbufferLoggerSession*> m_sessions;
};

class MessageActionBase:NonCopyable {
public:
   explicit
   MessageActionBase(uint32_t t):
      m_bb_location(t)
   {}

   virtual
   ~MessageActionBase()
   {}

   uint32_t
   bb_location(void) const
   {
      return m_bb_location;
   }

   virtual
   void
   action(int block_level, BatchbufferLoggerOutput &dst) const = 0;
private:
   uint32_t m_bb_location;
};

class MessageItem:
      public MessageActionBase {
public:
   MessageItem(uint32_t bb_location, const char *fmt, va_list vp);
   MessageItem(uint32_t bb_location, const char *fmt);

   virtual
   void
   action(int block_level, BatchbufferLoggerOutput &dst) const;
private:
   std::string m_text;
};

class ActivateCounter:
      public MessageActionBase {
public:
   explicit
   ActivateCounter(uint32_t bb_location,
                   std::set<GPUCommandCounter*> *dst,
                   GPUCommandCounter *counter):
      MessageActionBase(bb_location),
      m_dst(dst),
      m_counter(counter)
   {}

   virtual
   void
   action(int block_level, BatchbufferLoggerOutput &dst) const
   {
      m_dst->insert(m_counter);
   }
private:
   std::set<GPUCommandCounter*> *m_dst;
   GPUCommandCounter *m_counter;
};

class DeactivateCounter:
      public MessageActionBase {
public:
   explicit
   DeactivateCounter(uint32_t bb_location,
                   std::set<GPUCommandCounter*> *dst,
                   GPUCommandCounter *counter):
      MessageActionBase(bb_location),
      m_dst(dst),
      m_counter(counter)
   {}

   virtual
   void
   action(int block_level, BatchbufferLoggerOutput &dst) const
   {
      m_dst->erase(m_counter);
   }
private:
   std::set<GPUCommandCounter*> *m_dst;
   GPUCommandCounter *m_counter;
};

class ReleaseCounter:
      public MessageActionBase {
public:
   explicit
   ReleaseCounter(uint32_t bb_location,
                  std::set<GPUCommandCounter*> *dst,
                  GPUCommandCounter *counter):
      MessageActionBase(bb_location),
      m_dst(dst),
      m_counter(counter)
   {}

   virtual
   void
   action(int block_level, BatchbufferLoggerOutput &dst) const
   {
      m_dst->erase(m_counter);
      delete m_counter;
   }
private:
   std::set<GPUCommandCounter*> *m_dst;
   GPUCommandCounter *m_counter;
};

class ResetCounter:
      public MessageActionBase {
public:
   explicit
   ResetCounter(uint32_t bb_location,
                GPUCommandCounter *counter):
      MessageActionBase(bb_location),
      m_counter(counter)
   {}

   virtual
   void
   action(int block_level, BatchbufferLoggerOutput &dst) const
   {
      m_counter->reset();
   }
private:
   GPUCommandCounter *m_counter;
};

class PrintCounter:
      public MessageActionBase {
public:
   explicit
   PrintCounter(uint32_t bb_location, const char *txt,
                GPUCommandCounter *counter):
      MessageActionBase(bb_location),
      m_counter(counter),
      m_text(txt)
   {}

   virtual
   void
   action(int block_level, BatchbufferLoggerOutput &dst) const
   {
      if (dst) {
         dst.clear_block_stack(block_level);
         dst.begin_block(m_text.c_str());
         m_counter->print(dst);
         dst.end_block();
      }
   }
private:
   GPUCommandCounter *m_counter;
   std::string m_text;
};

/* A MessageActionList is a list of messages/actions to perform */
class MessageActionList {
public:
   void
   add_item(std::shared_ptr<MessageActionBase> p)
   {
      m_items.push_back(p);
   }

   const std::list<std::shared_ptr<MessageActionBase> >&
   items(void) const
   {
      return m_items;
   }

   void
   take_items(MessageActionList &src)
   {
      m_items.splice(m_items.begin(), src.m_items);
   }
private:
   std::list<std::shared_ptr<MessageActionBase> > m_items;
};

/* An APIStartCallMarker gives the details of an API call
 * together with "where" in the batchbuffer the API
 * call started.
 */
class APIStartCallMarker {
public:
   APIStartCallMarker(int call_id,
                      const char *api_call,
                      const char *api_call_details,
                      uint32_t t):
      m_call_id(call_id),
      m_api_call(api_call),
      m_api_call_details(api_call_details),
      m_start_bb_location(t)
   {}

   void
   emit(uint32_t next_entry_start_bb_location,
        BatchbufferLoggerOutput &dst, unsigned int top_level,
        BatchbufferDecoder *decoder, const GEMBufferObject *batchbuffer) const;

   uint32_t
   start_bb_location(void) const
   {
      return m_start_bb_location;
   }

   int
   call_id(void) const
   {
      return m_call_id;
   }

   void
   add_ioctl_log_entry(const std::string &entry)
   {
      m_ioctl_log.push_back(entry);
   }

   void
   add_item(std::shared_ptr<MessageActionBase> p)
   {
      m_items.add_item(p);
   }

   static
   void
   print_ioctl_log(const std::list<std::string> &ioctl_log,
                   BatchbufferLoggerOutput &dst);

private:
   /* the ID number for the call */
   int m_call_id;

   /* name of the API call */
   std::string m_api_call;

   /* details of the API call */
   std::string m_api_call_details;

   /* location in the batchbuffer at the time
    * the marker was made.
    */
   uint32_t m_start_bb_location;

   /* messages/actions from the driver made after the marker */
   MessageActionList m_items;

   /* additional log-messages that come from ioctl's */
   std::list<std::string> m_ioctl_log;
};

class GEMBufferObject:NonCopyable {
public:
   /* Value passed is the value AFTER the ioctl
    * DRM_IOCTL_I915_GEM_CREATE; the kernel passes
    * pack the struct modified
    */
   explicit
   GEMBufferObject(int fd, const struct drm_i915_gem_create &pdata);

   /* Value passed is the value AFTER the ioctl
    * DRM_IOCTL_I915_GEM_CREATE; the kernel passes
    * pack the struct modified
    */
   explicit
   GEMBufferObject(int fd, const struct drm_i915_gem_userptr &pdata);

   /* To be called -BEFORE- the ioctl DRM_IOCTL_GEM_CLOSE of
    * the GEM
    */
   ~GEMBufferObject();

   /* Handle to the GEM BO */
   uint32_t
   handle(void) const
   {
      return m_handle;
   }

   /* size of GEM BO in bytes */
   uint64_t
   size(void) const
   {
      return m_size;
   }

   /* If underlying GEM BO was created with DRM_IOCTL_I915_GEM_USERPTR,
    * then returns the CPU address of the underlying memory
    */
   const void*
   user_ptr(void) const
   {
      return m_user_ptr;
   }

   bool
   on_gtt(uint32_t ctx) const
   {
      return m_gpu_address.find(ctx) != m_gpu_address.end();
   }

   /* GPU address of GEM BO, note that until
    *  update_gpu_address() is called the value
    *  is 0, which is guaranteed to be incorrect.
    */
   uint64_t
   gpu_address_begin(uint32_t ctx) const
   {
      auto iter = m_gpu_address.find(ctx);
      assert(iter != m_gpu_address.end());
      return iter->second;
   }

   /* Gives the GPU address for the very end of the BO */
   uint64_t
   gpu_address_end(uint32_t ctx) const
   {
      return m_size + gpu_address_begin(ctx);
   }

   /* returns true if the GPU address changed and the old value as well*/
   std::pair<bool, uint64_t>
   update_gpu_address(uint32_t ctx, uint64_t new_gpu_address)
   {
      auto iter = m_gpu_address.find(ctx);
      std::pair<bool, uint64_t> return_value(false, ~uint64_t(0));

      if (iter == m_gpu_address.end()) {
         return_value.first = true;
         m_gpu_address[ctx] = new_gpu_address;
      } else if (iter->second != new_gpu_address) {
         return_value.first = true;
         return_value.second = iter->second;
         iter->second = new_gpu_address;
      }
      return return_value;
   }

   template<typename T = void>
   const T*
   cpu_mapped(void) const
   {
      return static_cast<const T*>(get_mapped());
   }

   /* if contents are mapped, unmap them; previous
    * return values of cpu_mapped() are then invalid.
    */
   void
   unmap(void) const;

   int
   pread_buffer(void *dst, uint64_t start, uint64_t sz) const;

private:
   const void*
   get_mapped(void) const;

   /* File descriptor of ioctl to make GEM BO */
   int m_fd;

   uint32_t m_handle;
   uint64_t m_size;
   const uint8_t *m_user_ptr;

   /* The buffer mapped; there is a danger that mapping
    * the buffer without sufficient cache flushing
    * will give incorrect data; on the other hand,
    * the gen_decoder interface wants raw pointers
    * from which to read. Let's hope that cache
    * flushing is not needed for reading the contents.
    */
   mutable void *m_mapped;

   /* the location in the GPU address space of the GEM
    * object, this is updated by the kernel in the
    * value drm_i915_gem_exec_object2::offset, because
    * PPGTT is per HW context, a given GEM can be in
    * multiple GTT's simutaneously, as such the
    * GPU address is keyed by by HW context ID.
    */
   std::map<uint32_t, uint64_t> m_gpu_address;
};

/* class to extract a value from a gen_field_iterator */
class GPUCommandFieldValue {
public:
   explicit
   GPUCommandFieldValue(const gen_field_iterator &iter);

   template<typename T>
   T
   value(void) const;

   /**
    * Returns the gen_type as indicated by the gen_field_iterator
    * used to constructor, value is an (unnamed) enumeration of
    * gen_type.
    */
   unsigned int
   type(void) const
   {
      return m_gen_type;
   }

private:
   /* enum values from the unnamed enum in gen_field::type::kind */
   unsigned int m_gen_type;

   union {
      /* for types GEN_TYPE_FLOAT, GEN_TYPE_UFIXED and GEN_TYPE_SFIXED */
      float f;

      /* for type GEN_TYPE_INT */
      int64_t i;

      /* for types GEN_TYPE_UNKNOWN, GEN_TYPE_UINT,
       * GEN_TYPE_ADDRESS, GEN_TYPE_OFFSET, GEN_TYPE_ENUM
       */
      uint64_t u;

      /* for GEN_TYPE_BOOL
       */
      bool b;
   } m_value;
};


/* Return results for getting the GEMBufferObject
 * and offset into the GEMBufferObject of a GPU
 * address
 */
class GPUAddressQuery {
public:
   GPUAddressQuery(void):
      m_gem_bo(nullptr),
      m_offset_into_gem_bo(-1)
   {}

   GEMBufferObject *m_gem_bo;
   uint64_t m_offset_into_gem_bo;
};

/* A GPUCommand is a location within a GEM BO
 * specifying where a GPU command is.
 */
class GPUCommand {
public:
   /* when saving GPUCommand's that set GPU state, we key
    * the value by the op-code of the GPU command.
    */
   typedef uint32_t state_key;

   /* what we do with the GPUCommand on absorbing it:
    *  - save the value as state and do not print it immediately
    *  - print it immediately and show current GPU state
    *  - print it immediately and do now show current GPU state
    */
   enum gpu_command_type_t {
      gpu_command_save_value_as_state_hw_context,
      gpu_command_save_value_as_state_not_hw_context,
      gpu_command_set_register,
      gpu_command_show_value_with_gpu_state,
      gpu_command_show_value_without_gpu_state,
   };

   /* only defined for gpu_decode_type_t values
    * gpu_save_value_as_state and gpu_show_value_with_gpu_state
    */
   enum gpu_pipeline_type_t {
      gpu_pipeline_compute,
      gpu_pipeline_gfx,
   };

   GPUCommand(void);

   /* if grp is nullptr, then read use spec and the contents
    * at the location to figure out what is the GPU command.
    */
   GPUCommand(const GEMBufferObject *q, uint64_t dword_offset,
              struct gen_spec *spec, struct gen_group *grp = nullptr,
              int override_size = -1);

   GPUCommand(const GPUAddressQuery &q, struct gen_spec *spec,
              struct gen_group *grp = nullptr,
              int override_size = -1);

   const uint32_t*
   contents_ptr(void) const
   {
      return m_contents;
   }

   uint32_t
   operator[](unsigned int I) const
   {
      assert(I < contents_size());
      return m_contents[I];
   }

   uint32_t
   content(unsigned int I) const
   {
      assert(I < contents_size());
      return m_contents[I];
   }

   unsigned int
   contents_size(void) const
   {
      return m_dword_length;
   }

   struct gen_group*
   inst(void) const
   {
      return m_inst;
   }

   const GEMBufferObject*
   gem_bo(void) const
   {
      return m_gem_bo;
   }

   uint64_t
   offset(void) const
   {
      return m_gem_bo_offset;
   }

   uint64_t
   dword_offset(void) const
   {
      return offset() / sizeof(uint32_t);
   }

   enum gpu_command_type_t
   gpu_command_type(void) const
   {
      return m_command_type;
   }

   enum gpu_pipeline_type_t
   gpu_pipeline_type(void) const
   {
      return m_pipeline_type;
   }

   /* read a GPU address from a location within the GPUCommand */
   uint64_t
   get_gpu_address(const BatchRelocs &relocs,
                   uint64_t dword_offset_from_cmd_start,
                   bool ignore_lower_12_bits = true) const;

   /* Sets up the GPUCommand to read data from an internal storage
    * instead of from the GEM BO.
    */
   void
   archive_data(const BatchRelocs &relocs);

   /* Returns true if and only if the GPUCommand is reading data
    * from internal storage instead of from the GEM BO.
    */
   bool
   is_archived(void) const
   {
      return m_archived_data.size() == m_dword_length;
   }

   /* Extract the value of a field from a GPUCommand, saving
    * the value in dst. Returns true on success and false
    * on failure.
    */
   template<typename T>
   bool
   extract_field_value(const char *pname, T *dst,
                       bool read_from_header = false) const;

private:
   static
   enum gpu_command_type_t
   get_gpu_command_type(struct gen_group *inst);

   static
   enum gpu_pipeline_type_t
   get_gpu_pipeline_type(struct gen_group *inst);

   void
   complete_init(uint32_t dword_offset, struct gen_spec *spec,
                 struct gen_group *grp, int override_size);

   const GEMBufferObject *m_gem_bo;
   uint64_t m_gem_bo_offset;
   struct gen_group *m_inst;
   const uint32_t *m_contents;
   unsigned int m_dword_length;
   enum gpu_command_type_t m_command_type;
   enum gpu_pipeline_type_t m_pipeline_type;
   std::vector<uint32_t> m_archived_data;
};

/* A significant amount of state on i965 depends deeply on other
 * portions of state for decoding. The biggest example being
 * the values in STATE_BASE_ADDRESS.
 */
class i965LatchState {
public:
   class per_stage_values {
   public:
      per_stage_values(void):
         m_binding_table_count(-1),
         m_sampler_count(-1)
      {}

      int m_binding_table_count;
      int m_sampler_count;
   };

   i965LatchState(void);

   void
   update_state(BatchbufferDecoder *decoder, BatchbufferLoggerOutput &poutput,
                const GPUCommand &q);

   /* Tracking STATE_BASE_ADDRESS */
   uint64_t m_general_state_base_address;
   uint64_t m_surface_state_base_address;
   uint64_t m_dynamic_state_base_address;
   uint64_t m_instruction_base_address;

   /* value derived from 3D_STATE_XS */
   int m_VIEWPORT_count;
   per_stage_values m_VS, m_HS, m_DS, m_GS, m_PS, m_CS;

private:
   void
   update_stage_values(BatchbufferDecoder *decoder,
                       BatchbufferLoggerOutput &poutput,
                       const GPUCommand &q, per_stage_values *dst);

   static
   void
   update_state_base_address_helper(const GPUCommand &q,
                                    const char *value_enabled_name,
                                    uint64_t *dst, const char *value_name);

   void
   update_state_base_address(BatchbufferDecoder *decoder,
                             BatchbufferLoggerOutput &poutput,
                             const GPUCommand &q);
};

/* A simple container to track the value of registers.
 */
class i965Registers {
public:
   i965Registers(void)
   {}

   void
   update_state(BatchbufferDecoder *decoder, BatchbufferLoggerOutput &poutput,
                const GPUCommand &q);

   void
   decode_contents(BatchbufferDecoder *decoder,
                   enum GPUCommand::gpu_pipeline_type_t pipeline,
                   BatchbufferLoggerOutput &poutput);

   void
   reset_state(void)
   {
      m_register_values.clear();
   }

private:
   /* register values are part of state, the key
    * to the map is the register offset and the value
    * is the value of the register.
    */
   std::map<uint32_t, uint32_t> m_register_values;
};

/* The execbuffer2 ioctls, (DRM_IOCTL_I915_GEM_EXECBUFFER2
 * and DRM_IOCTL_I915_GEM_EXECBUFFER2_WR) can pass a HW
 * context (via a uint32_t). When a driver uses a HW context,
 * it can avoid sending large amounts of state commands to
 * restore state. However, when we decode a batchbuffer,
 * we need to record HW state that impacts decoding
 * batchbuffers. The Bspec page to examine for what is
 * saved and restored in a HW context is at
 * gfxspecs.intel.com/Predator/Home/Index/20855
 */
class i965HWContextData {
public:
   explicit
   i965HWContextData(uint32_t ctx_id);
   ~i965HWContextData();

   uint32_t
   ctx_id(void) const
   {
      return m_ctx_id;
   }

   void
   decode_contents(BatchbufferDecoder *decoder,
                   enum GPUCommand::gpu_pipeline_type_t pipeline,
                   BatchbufferLoggerOutput &poutput);

   void
   update_state(BatchbufferDecoder *decoder, BatchbufferLoggerOutput &poutput,
                const GPUCommand &Q);

   void
   reset_state(void)
   {
      m_state.clear();
      m_registers.reset_state();
      m_latch_state = i965LatchState();
   }

   /* Batchbuffer decoding needs to examine and change
    * the values in i965LatchState when decoding some
    * elements of state.
    */
   i965LatchState m_latch_state;

private:
   uint32_t m_ctx_id;
   std::map<GPUCommand::state_key, GPUCommand> m_state;
   i965Registers m_registers;
};

class GPUState {
public:
   explicit
   GPUState(i965HWContextData *ctx):
      m_ctx_data(ctx)
   {
      assert(m_ctx_data);
   }

   void
   update_state(BatchbufferDecoder *decoder, BatchbufferLoggerOutput &poutput,
                const GPUCommand &Q);

   void
   decode_contents(BatchbufferDecoder *decoder,
                   enum GPUCommand::gpu_pipeline_type_t pipeline,
                   BatchbufferLoggerOutput &poutput);

   i965HWContextData&
   ctx(void) const
   {
      return *m_ctx_data;
   }

private:
   /* holder for state of the GW context */
   i965HWContextData *m_ctx_data;

   /* state that is not saved in the HW context */
   std::map<GPUCommand::state_key, GPUCommand> m_state;
   i965Registers m_registers;
};

/* A BatchRelocs tracks the relocation data reported back
 * from the kernel after an ioctl
 */
class BatchRelocs {
public:
   explicit
   BatchRelocs(gen_spec *spec):
      m_32bit_gpu_addresses(spec && gen_spec_get_gen(spec) < gen_make_gen(8, 0))
   {
   }

   /* Add relocation entries
    * \param exec_object from which to extract reloc data
    * \param tracker GEMBufferTracker to set GPU addresses
    * \param lut_references if non-null, then handle field
    *                       of each drm_i915_gem_relocation_entry
    *                       is an index into lut_references
    */
   void
   add_entries(const struct drm_i915_gem_exec_object2 &exec_object,
               GEMBufferObject *q, GEMBufferTracker *tracker,
               PerHWContext *ctx,
               const struct drm_i915_gem_exec_object2 *lut_references);

   void
   add_entry(const GEMBufferObject *gem,
             uint64_t offset_into_gem,
             uint64_t gpu_address)
   {
      m_relocs[gem][offset_into_gem] = gpu_address;
   }

   /* Write into dst any relocations found from the given GEM,
    * with dst representing the offset in -bytes- from the start
    * of the GEM.
    */
   void
   place_relocation_values_into_buffer(const GEMBufferObject *gem, uint64_t gem_bo_offset,
                                       std::vector<uint32_t> *dst) const;

   /* Decode/get a GPU address from a location in a GEMBufferObject
    *   - dword_offset in units of uint32_t's
    *   - if ignore_lower_12_bts is true, then low 12-bits of the
    *     passed gpu-address are ignored and the fetch is as if
    *     they are zero
    */
   uint64_t
   get_gpu_address(const GEMBufferObject *q, uint64_t dword_offset,
                   const uint32_t *p, bool ignore_lower_12_bits = true) const;


   void
   emit_reloc_data(BatchbufferLoggerOutput &poutput);

private:
   bool m_32bit_gpu_addresses;

   /* m_relocs[p] gives how to potentially reinterpret GPU addresses
    * when reading from buffer object p. That list is an std::map
    * keyed by offsets into p with values as the correct address
    * at that offset.
    */
   typedef std::map<uint64_t, uint64_t> reloc_map_of_gem_bo;
   typedef std::map<const GEMBufferObject*, reloc_map_of_gem_bo> reloc_map;
   reloc_map m_relocs;
};

/* A ShaderFileList acts a map from shaders to filenames (or
 * actual shader disassembly). A hash value is used as the key
 * of the map. If contents of a shader are not found, then a
 * new entry is made.
 */
class ShaderFileList:NonCopyable {
public:
   ShaderFileList(void):
      m_count(0)
   {}

   const char*
   filename(const void *shader, int pciid, struct gen_disasm *gen_disasm,
            const char *label);

   const char*
   disassembly(const void *shader, int pciid, struct gen_disasm *gen_disasm);

   void
   clear(void)
   {
      m_count = 0;
      m_files.clear();
   }

private:
   typedef std::array<unsigned char, 20> sha1_value;
   typedef std::pair<sha1_value, std::string> key_type;

   int m_count;
   std::map<key_type, std::string> m_files;
   std::map<sha1_value, std::string> m_disassembly;
};

/* A BatchbufferDecoder assists in the decoding the contents
 * of a batchbuffer, using the machinery in a GEMBufferTracker
 * to correctly read the contents of indirect state.
 */
class BatchbufferDecoder:NonCopyable {
public:
   enum decode_level_t {
      no_decode,
      instruction_decode,
      instruction_details_decode
   };

   enum print_reloc_level_t {
      print_reloc_nothing,
      print_reloc_gem_gpu_updates,
   };

   /* enumeration that gives what bit on shader decode */
   enum shader_decode_entry_t {
      shader_decode_vs,
      shader_decode_hs,
      shader_decode_ds,
      shader_decode_gs,
      shader_decode_ps_8,
      shader_decode_ps_16,
      shader_decode_ps_32,
      shader_decode_media_compute,

      shader_decode_entry_count,
   };

   BatchbufferDecoder(enum decode_level_t decode_level,
                      enum print_reloc_level_t print_reloc_level,
                      bool decode_shaders, bool organize_by_ioctls,
                      struct gen_spec *spec,
                      struct gen_disasm *dis,
                      int pciid,
                      GEMBufferTracker *tracker,
                      bool is_post_call, BatchRelocs &relocs,
                      GPUCommandCounter *gpu_command_counter,
                      ShaderFileList *shader_filelist,
                      struct drm_i915_gem_execbuffer2 *execbuffer2);

   ~BatchbufferDecoder();

   static
   void
   handle_batchbuffer_contents(BatchbufferDecoder *decoder, BatchbufferLoggerOutput &dst,
                               const GEMBufferObject *batchbuffer, uint32_t start, uint32_t end);

   void
   decode_gpu_command(BatchbufferLoggerOutput &poutput, const GPUCommand &q);

   const GEMBufferTracker&
   tracker(void) const
   {
      return *m_tracker;
   }

   const PerHWContext&
   ctx(void) const
   {
      return *m_ctx;
   }

   const GEMBufferObject*
   batchbuffer(void)
   {
      return m_batchbuffer;
   }

   const BatchRelocs&
   relocs(void) const
   {
      return m_relocs;
   }

   BatchbufferLog*
   batchbuffer_log(void)
   {
      return m_batchbuffer_log;
   }

   struct gen_spec*
   spec(void) const
   {
      return m_spec;
   }

   void
   emit_log(BatchbufferLoggerOutput &file, int count);

private:
   class DetailedDecoder:NonCopyable
   {
   public:
      static
      void
      decode(BatchbufferDecoder *decoder, BatchbufferLoggerOutput &poutput,
             const GPUCommand &data);

   private:
      typedef void (BatchbufferDecoder::*fcn)(BatchbufferLoggerOutput &poutput,
                                              const GPUCommand &data);

      DetailedDecoder(void);

      /* keyed by op-code */
      std::map<uint32_t, fcn> m_elements;
   };

   void
   emit_execbuffer2_details(BatchbufferLoggerOutput &poutput);

   void
   absorb_batchbuffer_contents(BatchbufferLoggerOutput &poutput,
                               const GEMBufferObject *batchbuffer,
                               unsigned int start_dword, unsigned int end_dword);

   void
   build_driver_values(void);

   void
   handle_batchbuffer_start(BatchbufferLoggerOutput &dst,
                            const GPUCommand &gpu_command);

   void
   decode_gen_group(BatchbufferLoggerOutput &poutput,
                    const GEMBufferObject *q, uint64_t offset,
                    const uint32_t *p, struct gen_group *inst);

   void
   decode_gpu_execute_command(BatchbufferLoggerOutput &poutput,
                              const GPUCommand &q);

   void
   process_gpu_command(BatchbufferLoggerOutput &poutput,
                       const GPUCommand &q);

   void
   decode_pointer_helper(BatchbufferLoggerOutput &poutput,
                         struct gen_group *g, uint64_t gpu_address);

   void
   decode_pointer_helper(BatchbufferLoggerOutput &poutput,
                         const char *instruction_name,
                         uint64_t gpu_address);

   void
   decode_shader(BatchbufferLoggerOutput &poutput,
                 enum shader_decode_entry_t tp, uint64_t gpu_address);

   void
   decode_3dstate_binding_table_pointers(BatchbufferLoggerOutput &poutput,
                                         const std::string &label, uint32_t offset,
                                         int cnt);

   void
   decode_3dstate_sampler_state_pointers_helper(BatchbufferLoggerOutput &poutput,
                                                uint32_t offset, int cnt);

   void
   decode_media_interface_descriptor_load(BatchbufferLoggerOutput &poutput,
                                          const GPUCommand &data);

   void
   decode_3dstate_xs(BatchbufferLoggerOutput &poutput,
                     const GPUCommand &data);

   void
   decode_3dstate_ps(BatchbufferLoggerOutput &poutput,
                     const GPUCommand &data);

   void
   decode_3dstate_constant(BatchbufferLoggerOutput &poutput,
                           const GPUCommand &data);

   void
   decode_3dstate_binding_table_pointers_vs(BatchbufferLoggerOutput &poutput,
                                            const GPUCommand &data);

   void
   decode_3dstate_binding_table_pointers_ds(BatchbufferLoggerOutput &poutput,
                                            const GPUCommand &data);

   void
   decode_3dstate_binding_table_pointers_hs(BatchbufferLoggerOutput &poutput,
                                            const GPUCommand &data);

   void
   decode_3dstate_binding_table_pointers_gs(BatchbufferLoggerOutput &poutput,
                                            const GPUCommand &data);

   void
   decode_3dstate_binding_table_pointers_ps(BatchbufferLoggerOutput &poutput,
                                            const GPUCommand &data);

   void
   decode_3dstate_sampler_state_pointers_vs(BatchbufferLoggerOutput &poutput,
                                            const GPUCommand &data);

   void
   decode_3dstate_sampler_state_pointers_gs(BatchbufferLoggerOutput &poutput,
                                            const GPUCommand &data);

   void
   decode_3dstate_sampler_state_pointers_hs(BatchbufferLoggerOutput &poutput,
                                            const GPUCommand &data);

   void
   decode_3dstate_sampler_state_pointers_ds(BatchbufferLoggerOutput &poutput,
                                            const GPUCommand &data);

   void
   decode_3dstate_sampler_state_pointers_ps(BatchbufferLoggerOutput &poutput,
                                            const GPUCommand &data);

   void
   decode_3dstate_sampler_state_pointers_gen6(BatchbufferLoggerOutput &poutput,
                                              const GPUCommand &data);

   void
   decode_3dstate_viewport_state_pointers_cc(BatchbufferLoggerOutput &poutput,
                                             const GPUCommand &data);

   void
   decode_3dstate_viewport_state_pointers_sf_clip(BatchbufferLoggerOutput &poutput,
                                                  const GPUCommand &data);

   void
   decode_3dstate_blend_state_pointers(BatchbufferLoggerOutput &poutput,
                                       const GPUCommand &data);

   void
   decode_3dstate_cc_state_pointers(BatchbufferLoggerOutput &poutput,
                                    const GPUCommand &data);

   void
   decode_3dstate_scissor_state_pointers(BatchbufferLoggerOutput &poutput,
                                         const GPUCommand &data);

   enum decode_level_t m_decode_level;
   enum print_reloc_level_t m_print_reloc_level;
   bool m_decode_shaders, m_organize_by_ioctls;
   struct gen_spec *m_spec;
   struct gen_disasm *m_gen_disasm;
   int m_pci_id;
   GEMBufferTracker *m_tracker;
   GPUCommandCounter *m_gpu_command_counter;
   ShaderFileList *m_shader_filelist;
   const GEMBufferObject *m_batchbuffer;
   BatchbufferLog *m_batchbuffer_log;
   std::vector<GEMBufferObject*> m_buffers;
   bool m_reloc_handles_are_indices;
   PerHWContext *m_ctx;
   GPUState m_gpu_state;
   BatchRelocs &m_relocs;
   struct drm_i915_gem_execbuffer2 *m_execbuffer2;
};

/* The type to hold the log associated to a single batchbuffer */
class BatchbufferLog {
public:
   BatchbufferLog(int fd, const void *driver_data, uint32_t h)
   {
      m_src.gem_bo = h;
      m_src.fd = fd;
      m_src.driver_data = driver_data;
   }

   BatchbufferLog(void)
   {
      m_src.gem_bo = 0;
      m_src.fd = -1;
      m_src.driver_data = nullptr;
   }

   const struct i965_logged_batchbuffer*
   src(void) const
   {
      return &m_src;
   }

   void
   add_item(std::shared_ptr<MessageActionBase>);

   void //only emits API markers
   emit_log(BatchbufferLoggerOutput &dst) const;

   void //emits API markers with batchbuffer decoding
   emit_log(BatchbufferDecoder *decoder, BatchbufferLoggerOutput &file,
            const GEMBufferObject *batchbuffer,
            uint32_t batchbuffer_start, uint32_t batchbuffer_len,
            int bb_id) const;

   void
   add_call_marker(BatchbufferLog &dummy, unsigned int call_id,
                   const char *fcn_name, const char *call_detailed,
                   uint32_t bb_location)
   {
      if (this != &dummy) {
         m_prints_from_dummy.splice(m_prints_from_dummy.end(),
                                    dummy.m_prints);
         m_pre_print_items.take_items(dummy.m_pre_print_items);
      }
      APIStartCallMarker ap(call_id, fcn_name, call_detailed,
                            bb_location);
      m_prints.push_back(ap);
   }

   uint32_t
   first_api_call_id(void) const
   {
      assert(!m_prints.empty() || !m_prints_from_dummy.empty());
      return (!m_prints_from_dummy.empty()) ?
         m_prints_from_dummy.front().call_id() :
         m_prints.front().call_id();
   }

   uint32_t
   last_api_call_id(void) const
   {
      assert(!m_prints.empty() || !m_prints_from_dummy.empty());
      return (!m_prints.empty()) ?
         m_prints.back().call_id() :
         m_prints_from_dummy.back().call_id();
   }

   void
   clear(void)
   {
      m_prints.clear();
      m_prints_from_dummy.clear();
   }

   bool
   empty(void) const
   {
      return m_prints.empty() && m_prints_from_dummy.empty();
   }

   void
   add_ioctl_log_entry(const std::string &entry);

   void
   absorb_log(BatchbufferLog *from)
   {
      m_prints.splice(m_prints.begin(), from->m_prints);
      m_prints_from_dummy.splice(m_prints_from_dummy.begin(),
                                 from->m_prints_from_dummy);
      m_orphan_ioctl_log_entries.splice(m_orphan_ioctl_log_entries.begin(),
                                        from->m_orphan_ioctl_log_entries);
   }

private:
   friend class GEMBufferTracker;

   /* src parameters of the BatchbufferLog object */
   struct i965_logged_batchbuffer m_src;

   /* Messages added via i965_batchbuffer_logger::add_item()
    * before any APIStartCallMarker are added to the log
    */
   MessageActionList m_pre_print_items;

   /* API markers of the batchbuffer */
   std::list<APIStartCallMarker> m_prints;

   /* For the markers emmitted when there is not active
    * batchbuffer land in BatchbufferLogger::m_dummy.
    * The first time BatchbufferLogger has a valid batch
    * buffer, the merkers of m_dummy are spliced onto
    * those batchbuffer's log here.
    */
   std::list<APIStartCallMarker> m_prints_from_dummy;

   /* when an ioctl log entry is added but there are no
    * APIStartCallMarker to which to add it.
    */
   std::list<std::string> m_orphan_ioctl_log_entries;
};

class PerHWContext:
      private NonCopyable,
      public i965HWContextData {
public:
   explicit
   PerHWContext(uint32_t id);

   ~PerHWContext();

   bool
   update_gem_bo_gpu_address(GEMBufferObject *gem, uint64_t new_address);

   /* Return what GEM BO and offset into
    * that GEM BO for a given GPU address.
    */
   GPUAddressQuery
   get_memory_at_gpu_address(uint64_t gpu_address) const;

   /* Use kernel interface pread to read contents */
   int
   pread_buffer(void *dst, uint64_t gpu_address, uint64_t size) const;

   /* Get mapped of a GEM BO given from a GPU Address */
   template<typename T>
   const T*
   cpu_mapped(uint64_t gpu_address, GPUAddressQuery *q = nullptr);

   void
   drop_gem(GEMBufferObject *p);

private:
   /* GEM BO's keyed by the GPU address of the end of the GEM BO*/
   std::map<uint64_t, GEMBufferObject*> m_gem_bos_by_gpu_address_end;
};

class GEMBufferTracker:NonCopyable {
public:
   explicit
   GEMBufferTracker(int fd);

   ~GEMBufferTracker();

   /* Add a GEM BO, to be called after the ioctl
    * DRM_IOCTL_I915_GEM_CREATE returns with the
    * kernel modified drm_i915_gem_create value
    */
   void
   add_gem_bo(const struct drm_i915_gem_create &pdata);

   /* Add a GEM BO, to be called after the ioctl
    * DRM_IOCTL_I915_GEM_USERPTR returns with the
    * kernel modified drm_i915_gem_userptr value
    */
   void
   add_gem_bo(const struct drm_i915_gem_userptr &pdata);

   /* remove a GEM BO from tracking */
   void
   remove_gem_bo(uint32_t h);

   /* Fetch a GEMBufferObject given a GEM handle */
   GEMBufferObject*
   fetch_gem_bo(uint32_t h) const;

   /* Add a new HW GEM context for tracking */
   void
   add_hw_context(const struct drm_i915_gem_context_create &create);

   /* remove a HW GEM context for tracking */
   void
   remove_hw_context(const struct drm_i915_gem_context_destroy &destroy);

   /* fetch a GEM HW context from a handle */
   PerHWContext*
   fetch_hw_context(uint32_t h);

   /* to be called just after the ioctl DRM_IOCTL_I915_GEM_EXECBUFFER2
    * or DRM_IOCTL_I915_GEM_EXECBUFFER2_WR is issued passing the GEM BO
    * list modified by the kernel; returns what GEMBufferObject had the
    * GEM handle and if the GPU address did get changed
    */
   std::pair<bool, GEMBufferObject*>
   update_gem_bo_gpu_address(uint32_t ctx,
                             const struct drm_i915_gem_exec_object2 *p);

   /* to be called just after the ioctl DRM_IOCTL_I915_GEM_EXECBUFFER2
    * or DRM_IOCTL_I915_GEM_EXECBUFFER2_WR is issued passing the GEM BO
    * list modified by the kernel; returns what GEMBufferObject had the
    * GEM handle and if the GPU address did get changed
    */
   std::pair<bool, GEMBufferObject*>
   update_gem_bo_gpu_address(PerHWContext *ctx,
                             const struct drm_i915_gem_exec_object2 *p);

   /* Instead of calling DRM_IOCTL_I915_GEM_EXECBUFFER2 ioctl
    * and using the returned values, issue a DRM_IOCTL_I915_GEM_EXECBUFFER2
    * ioctl whose command is just a BATCHBUFFER_END, but it has
    * all the reloc reqests that the passed drm_i915_gem_execbuffer2
    * has.
    */
   void
   update_gem_bo_gpu_addresses(const struct drm_i915_gem_execbuffer2 *p,
                               unsigned int req, BatchRelocs *out_relocs);

   /* Fetch (or create) a BatchbufferLog given a
    * GEM handle and an opaque pointer provided by the
    * driver for a batchbuffer.
    */
   BatchbufferLog*
   fetch_or_create(const void *opaque_bb, uint32_t gem_handle);

   /* Fetch a BatchbufferLog given a GEM handle, if
    * no BatchbufferLog exists, then return nullptr
    */
   BatchbufferLog*
   fetch(uint32_t gem_handle);

   /* remove a BatchbufferLog from tracking */
   void
   remove_batchbuffer_log(const BatchbufferLog *q);

   bool
   fd_has_exec_capture(void) const
   {
      return m_fd_has_exec_capture;
   }

private:
   int m_fd;
   bool m_fd_has_exec_capture;

   /* GEM BO's keyed by DRM handle */
   std::map<uint32_t, GEMBufferObject*> m_gem_bos_by_handle;

   /* HW contexts keyed by DRM handle */
   std::map<uint32_t, PerHWContext*> m_hw_contexts;

   /* dummy HW context for execbuffer calls without hw
    * context, GPU state tracking is reset after each
    * call
    */
   PerHWContext m_dummy_hw_ctx;

   /* backing storage for the logs, keyed by
    * batchbuffer DRM handle
    */
   std::map<uint32_t, BatchbufferLog> m_logs;
};

/* A ManagedGEMBuffer is a conveniance class
 * to creating and destroying gem_bo's via
 * the kernel interface
 */
class ManagedGEMBuffer:NonCopyable {
public:
   explicit
   ManagedGEMBuffer(int fd, __u64 sz);

   ~ManagedGEMBuffer();

   __u32
   handle(void) const
   {
      return m_handle;
   }

   __u64
   size(void) const
   {
      return m_sz;
   }

   template<typename T>
   T*
   map(void)
   {
      return static_cast<T*>(map_implement());
   }

   void
   unmap(void);
private:
   void*
   map_implement(void);

   int m_fd;
   __u32 m_handle;
   __u64 m_sz;
   void *m_mapped;
};

class BatchbufferLogger:
      public i965_batchbuffer_logger,
      public i965_batchbuffer_logger_app,
      NonCopyable {
public:
   static
   BatchbufferLogger*
   acquire(void);

   static
   void
   release(void);

   static
   int
   local_drm_ioctl(int fd, unsigned long request, void *argp);

   void
   set_pci_id(int pci_id);

   void
   set_driver_funcs(i965_logged_batchbuffer_state f1,
                    i965_active_batchbuffer f2);

   /* if returns a non-null value, that object should be
    * "added" to the execbuffer2 command as an exec_object2
    * with the flag value EXEC_OBJECT_CAPTURE.
    */
   ManagedGEMBuffer*
   pre_process_ioctl(int fd, unsigned long request, void *argp);

   void
   post_process_ioctl(int ioctl_return_code, int fd, unsigned long request, void *argp);

private:
   BatchbufferLogger(void);
   ~BatchbufferLogger();

   GEMBufferTracker*
   gem_buffer_tracker(int fd);

   /* Returns nullptr if fd is -1 or if the
    * GEMBufferTracker associated to the fd
    * does not have a BatchbufferLog of
    * the given gem_bo
    */
   BatchbufferLog*
   fetch_batchbuffer_log(int fd, uint32_t gem_bo);

   /* if fd is -1, then returns the dummy BatchbufferLog,
    * otherwise fetches_or_crates a BatchbufferLog from
    * the fields of the passed batchbuffer
    */
   BatchbufferLog*
   fetch_or_create_batchbuffer_log(const struct i965_logged_batchbuffer*);

   /* Calls m_active_batchbuffer to get the value of
    * the active batchbuffer and uses that.
    */
   BatchbufferLog*
   fetch_or_create_batchbuffer_log(void);

   void
   remove_batchbuffer_log(const BatchbufferLog *q);

   /* called on each call to release(), in case application fails to
    * call release_app() or if it fails to disconnect from X elegantly
    * (which means that release_driver() is not called from Mesa).
    */
   void
   emit_total_stats(void);

   void
   print_ioctl_message(const std::string &msg);

   static
   void
   clear_batchbuffer_log_fcn(struct i965_batchbuffer_logger*,
                             int fd, uint32_t gem_bo);

   static
   void
   migrate_batchbuffer_fcn(struct i965_batchbuffer_logger*,
                           const struct i965_logged_batchbuffer *from,
                           const struct i965_logged_batchbuffer *to);

   static
   void
   add_message_fcn(struct i965_batchbuffer_logger *logger,
                   const struct i965_logged_batchbuffer *dst,
                   const char *fmt);

   static
   void
   release_driver_fcn(struct i965_batchbuffer_logger *pthis);

   static
   struct i965_batchbuffer_counter
   create_counter_fcn(struct i965_batchbuffer_logger *pthis,
                      const char *filename);

   static
   void
   activate_counter_fcn(struct i965_batchbuffer_logger *logger,
                        struct i965_batchbuffer_counter counter);

   static
   void
   deactivate_counter_fcn(struct i965_batchbuffer_logger *logger,
                          struct i965_batchbuffer_counter counter);

   static
   void
   print_counter_fcn(struct i965_batchbuffer_logger *logger,
                     struct i965_batchbuffer_counter counter,
                     const char *label);

   static
   void
   reset_counter_fcn(struct i965_batchbuffer_logger *logger,
                     struct i965_batchbuffer_counter counter);

   static
   void
   release_counter_fcn(struct i965_batchbuffer_logger *logger,
                       struct i965_batchbuffer_counter counter);

   static
   void
   pre_call_fcn(struct i965_batchbuffer_logger_app *pthis,
                unsigned int call_id,
                const char *call_detailed,
                const char *fcn_name);

   static
   void
   post_call_fcn(struct i965_batchbuffer_logger_app *pthis,
                 unsigned int call_id);

   static
   struct i965_batchbuffer_logger_session
   begin_file_session_fcn(struct i965_batchbuffer_logger_app *pthis,
                          const char *filename);

   static
   struct i965_batchbuffer_logger_session
   begin_session_fcn(struct i965_batchbuffer_logger_app *pthis,
                     const struct i965_batchbuffer_logger_session_params *params);

   static
   void
   end_session_fcn(struct i965_batchbuffer_logger_app *pthis,
                   struct i965_batchbuffer_logger_session);

   static
   void
   release_app_fcn(struct i965_batchbuffer_logger_app *pthis);

   static
   uint32_t
   default_batchbuffer_state_fcn(const struct i965_logged_batchbuffer *st)
   {
      return 0;
   }

   static
   void
   default_active_batchbuffer_fcn(struct i965_logged_batchbuffer *st)
   {
      st->fd = -1;
      st->gem_bo = 0;
      st->driver_data = nullptr;
   }

   /* derived fron enviromental string */
   enum BatchbufferDecoder::decode_level_t m_decode_level;
   enum BatchbufferDecoder::print_reloc_level_t m_print_reloc_level;
   bool m_decode_shaders, m_organize_by_ioctls;
   bool m_process_execbuffers_before_ioctl;
   bool m_emit_capture_execobj_batchbuffer_identifier;

   /* from driver */
   i965_logged_batchbuffer_state m_batchbuffer_state;
   i965_active_batchbuffer m_active_batchbuffer;
   int m_pci_id;

   /* derived data from m_pci_id */
   struct gen_device_info m_dev_info;
   struct gen_spec *m_gen_spec;
   struct gen_disasm *m_gen_disasm;

   /* unique creation ID for the very rare case
    * where during the lifetime of an application
    * that more than one BatchbufferLogger is made
    */
   unsigned int m_creation_ID;

   /* GEM buffer tracking, keyed by file descriptor */
   std::map<int, GEMBufferTracker*> m_gem_buffer_trackers;

   /* number of ioctls (to guaranteee unique ID's) */
   int m_number_ioctls;

   ShaderFileList m_shader_filelist;

   /* thread safety guaranteed by std */
   std::mutex m_mutex;

   /* special dummy batchbuffer; markers are added
    * to it if there is no active batchbuffer, the
    * first time we get an active batchbuffer, the
    * markers on dummy are given to the
    * BatchbufferLog associated to it.
    */
   BatchbufferLog m_dummy;

   GPUCommandCounter m_gpu_command_counter;

   /* list of active counters, coutners are made active and
    * active by commands in a BatchbufferLog.
    */
   std::set<GPUCommandCounter*> m_active_counters;

   /* collction of sessions */
   BatchbufferLoggerOutput m_output;
};

/* The entire purpose of ForceExistanceOfBatchbufferLogger
 * is to guarantee the existance of a BatchbufferLogger.
 * We do NOT make the BatchbufferLogger a static variable
 * (be it at global scope or in a function scope) because
 * there are no hard gaurantees of when its dtor will be
 * called with respect to other static objects across other
 * files. So instead, the methods BatchbufferLogger::acquire()
 * and BatchbufferLogger::release() implement creation and
 * reference counting. If an application (or even driver)
 * has that they release the BatchbufferLogger in a dtor
 * of a static object (or use it for that matter), the reference
 * counting guarnatees that the BatchbufferLogger will
 * exist (i.e. not have its dtor called). The class
 * ForceExistanceOfBatchbufferLogger is to make sure that
 * a BatchbufferLogger exists by having acquire() called on
 * ctor and release() called on dtor. The object itself can
 * then be instanced statically.
 */
class ForceExistanceOfBatchbufferLogger {
public:
   ForceExistanceOfBatchbufferLogger(void)
   {
      BatchbufferLogger::acquire();
   }

   ForceExistanceOfBatchbufferLogger(const ForceExistanceOfBatchbufferLogger&)
   {
      BatchbufferLogger::acquire();
   }

   ~ForceExistanceOfBatchbufferLogger(void)
   {
      BatchbufferLogger::release();
   }
};

} //namespace

////////////////////////////////
// GPUCommandCounterValue methods
GPUCommandCounterValue::
GPUCommandCounterValue(void):
   m_inst(nullptr),
   m_accumulated_count(0),
   m_accumulated_length(0)
{}

GPUCommandCounterValue::
GPUCommandCounterValue(const GPUCommandCounterValue &from,
                       const GPUCommandCounterValue &to):
   m_inst(to.m_inst)
{
   assert(to.m_inst == from.m_inst || !to.m_inst || !from.m_inst);
   m_accumulated_count = diff_helper(from.m_accumulated_count,
                                     to.m_accumulated_count);
   m_accumulated_length = diff_helper(from.m_accumulated_length,
                                      to.m_accumulated_length);
}

void
GPUCommandCounterValue::
increment(const GPUCommand &q)
{
   assert(!m_inst || m_inst == q.inst());
   m_inst = q.inst();
   ++m_accumulated_count;
   m_accumulated_length += q.contents_size();
}

///////////////////////////////////////
// PipeControlCounterValue methods
PipeControlCounterValue::
PipeControlCounterValue(void):
   m_base(),
   m_counts()
{}

PipeControlCounterValue::
PipeControlCounterValue(const PipeControlCounterValue &from,
                        const PipeControlCounterValue &to):
   m_base(from.m_base, to.m_base)
{
   for (unsigned int i = 0; i < number_entries; ++i) {
      m_counts[i] = diff_helper(from.m_counts[i], to.m_counts[i]);
   }
}

void
PipeControlCounterValue::
increment(const GPUCommand &q)
{
   m_base.increment(q);
   for (unsigned int i = 0; i < number_entries; ++i) {
      bool enabled(false);
      if (q.extract_field_value(PipeControlFields[i], &enabled) && enabled) {
         ++m_counts[i];
      }
   }
}

bool
PipeControlCounterValue::
counts_non_zero(void) const
{
   bool return_value(false);

   for(unsigned int i = 0; i < number_entries && !return_value; ++i) {
      return_value = (m_counts[i] > 0);
   }
   return return_value;
}

/////////////////////////////////////////
// GPUCommandCounter methods
GPUCommandCounter::
GPUCommandCounter(const GPUCommandCounter &from,
                  const GPUCommandCounter &to):
   m_pipe_value(from.m_pipe_value, to.m_pipe_value)
{
   m_total_length = diff_helper(from.m_total_length,
                                to.m_total_length);
   m_total_count = diff_helper(from.m_total_count,
                               to.m_total_count);
   m_number_batchbuffers = diff_helper(from.m_number_batchbuffers,
                                       to.m_number_batchbuffers);

   /* iterate through to and from together to get differences */
   for(auto to_iter = to.m_values.begin(), to_end = to.m_values.end(),
          from_iter = from.m_values.begin(), from_end = from.m_values.end();
       to_iter != to_end; ++to_iter) {
      GPUCommandCounterValue value;

      /* advance from_iter so that from_iter->first is not less than to->first */
      for (; from_iter != from_end && from_iter->first < to_iter->first; ++from_iter) {}

      if (from_iter != from_end && from_iter->first == to_iter->first) {
         value = GPUCommandCounterValue(from_iter->second, to_iter->second);
      } else {
         value = to_iter->second;
      }

      if (value.non_zero()) {
         m_values[to_iter->first] = value;
      }
   }
}

void
GPUCommandCounter::
increment(const GPUCommand &q)
{
   struct gen_group *inst;
   inst = q.inst();

   ++m_total_count;
   m_total_length += q.contents_size();

   if (!inst) {
      return;
   }

   if (gen_group_get_opcode(inst) == _3DSTATE_PIPE_CONTROL) {
      m_pipe_value.increment(q);
   } else {
      m_values[gen_group_get_opcode(inst)].increment(q);
   }
}

void
GPUCommandCounter::
print(BatchbufferLoggerOutput &dst)
{
   dst.print_value("Number Batchbuffers", "%d", number_batchbuffers());
   dst.print_value("Sum Number Commands", "%d", total_count());
   dst.print_value("Sum Number Command DWORDS", "%d", total_length());
   for (const auto &v : m_values) {
      dst.begin_block(gen_group_get_name(v.second.inst()));
      dst.print_value("Count", "%d", v.second.accumulated_count());
      dst.print_value("Total dwords", "%d", v.second.accumulated_length());
      dst.end_block();
   }

   if (m_pipe_value.non_zero()) {
      dst.begin_block(gen_group_get_name(m_pipe_value.inst()));
      dst.print_value("Count", "%d", m_pipe_value.accumulated_count());
      dst.print_value("Total dwords", "%d", m_pipe_value.accumulated_length());
      if (m_pipe_value.counts_non_zero()) {
         dst.begin_block("Individual PIPE_CONTROL counts");
         for (unsigned int i = 0; i < PipeControlCounterValue::number_entries; ++i) {
               dst.print_value(PipeControlFields[i], "%d", m_pipe_value[i]);
         }
      }
      dst.end_block(); //"Individual PIPE_CONTROL counts"
      dst.end_block(); //gen_group_get_name
   }
}

//////////////////////////////////
// BatchbufferLoggerSession methods
BatchbufferLoggerSession::
BatchbufferLoggerSession(const struct i965_batchbuffer_logger_session_params &params,
                         const GPUCommandCounter *counter,
                         bool add_begin_logging_block):
   m_add_begin_logging_block(add_begin_logging_block),
   m_params(params),
   m_counter(counter),
   m_at_time_of_open(*counter)
{
   static const char *begin = "Logging Begin";
   if (m_add_begin_logging_block) {
      add_element(I965_BATCHBUFFER_LOGGER_MESSAGE_BLOCK_BEGIN,
                  begin, std::strlen(begin),
                  nullptr, 0);
   }
}

BatchbufferLoggerSession::
~BatchbufferLoggerSession()
{
   if (m_add_begin_logging_block) {
      add_element(I965_BATCHBUFFER_LOGGER_MESSAGE_BLOCK_END,
                  nullptr, 0, nullptr, 0);
   }

   GPUCommandCounter delta(m_at_time_of_open, *m_counter);
   BatchbufferLoggerOutput tmp;

   tmp.add_session(this);
   tmp.begin_block("Counter Stats");
   delta.print(tmp);
   tmp.end_block();
   tmp.remove_session(this);

   m_params.close(m_params.client_data);
}

void
BatchbufferLoggerSession::
add_element(enum i965_batchbuffer_logger_message_type_t tp,
            const void *name, uint32_t name_length,
            const void *value, uint32_t value_length)
{
   m_params.write(m_params.client_data, tp,
                  name, name_length,
                  value, value_length);
}

void
BatchbufferLoggerSession::
write_file(void *client_data,
           enum i965_batchbuffer_logger_message_type_t tp,
           const void *name, uint32_t name_length,
           const void *value, uint32_t value_length)
{
   struct i965_batchbuffer_logger_header hdr;
   std::FILE *file;

   file = static_cast<std::FILE*>(client_data);
   hdr.type = tp;
   hdr.name_length = name_length;
   hdr.value_length = value_length;

   std::fwrite(&hdr, sizeof(hdr), 1, file);
   if (name_length > 0) {
      std::fwrite(name, sizeof(char), name_length, file);
   }
   if (value_length > 0) {
      std::fwrite(value, sizeof(char), value_length, file);
   }
}

void
BatchbufferLoggerSession::
flush_file(void *client_data, unsigned int id)
{
   std::FILE *file;

   (void)id;
   file = static_cast<std::FILE*>(client_data);
   std::fflush(file);
}

void
BatchbufferLoggerSession::
close_file(void *client_data)
{
   std::FILE *file;

   file = static_cast<std::FILE*>(client_data);
   std::fclose(file);
}

/////////////////////////////////
//BatchbufferLoggerOutput methods
BatchbufferLoggerOutput::
BatchbufferLoggerOutput(void)
{}

BatchbufferLoggerOutput::
~BatchbufferLoggerOutput()
{
   assert(m_sessions.empty());
}

void
BatchbufferLoggerOutput::
close_all_sessions(void)
{
   clear_block_stack(0);
   for (BatchbufferLoggerSession *s : m_sessions) {
      delete s;
   }
   m_sessions.clear();
}

void
BatchbufferLoggerOutput::
add_session(BatchbufferLoggerSession *p)
{
   for (auto iter = m_block_stack.begin(); iter != m_block_stack.end(); ++iter) {
      p->add_element(I965_BATCHBUFFER_LOGGER_MESSAGE_BLOCK_BEGIN,
                     iter->first, iter->second);
   }
   m_sessions.insert(p);
}

bool
BatchbufferLoggerOutput::
remove_session(BatchbufferLoggerSession *p)
{
   std::set<BatchbufferLoggerSession*>::iterator iter;
   iter = m_sessions.find(p);
   if (iter != m_sessions.end()) {
      for (unsigned int i = 0, endi = m_block_stack.size(); i < endi; ++i) {
         p->add_element(I965_BATCHBUFFER_LOGGER_MESSAGE_BLOCK_END,
                        nullptr, 0, nullptr, 0);
      }
      m_sessions.erase(p);
   }

   return iter != m_sessions.end();
}

void
BatchbufferLoggerOutput::
pre_execbuffer2_ioctl(unsigned int id)
{
   for (BatchbufferLoggerSession *s : m_sessions) {
      s->pre_execbuffer2_ioctl(id);
   }
}

void
BatchbufferLoggerOutput::
post_execbuffer2_ioctl(unsigned int id)
{
   for (BatchbufferLoggerSession *s : m_sessions) {
      s->post_execbuffer2_ioctl(id);
   }
}

void
BatchbufferLoggerOutput::
begin_block(const char *txt)
{
   uint32_t name_length(std::strlen(txt));
   for (BatchbufferLoggerSession *s : m_sessions) {
      s->add_element(I965_BATCHBUFFER_LOGGER_MESSAGE_BLOCK_BEGIN,
                     txt, name_length, nullptr, 0);
   }
   u8vector name(txt, txt + name_length);
   m_block_stack.push_back(block_stack_entry(name, u8vector()));
}

void
BatchbufferLoggerOutput::
begin_block_value(const char *txt, const char *fmt, ...)
{
   va_list args;
   va_start(args, fmt);
   vbegin_block_value(txt, fmt, args);
   va_end(args);
}

void
BatchbufferLoggerOutput::
vbegin_block_value(const char *txt, const char *fmt, va_list va)
{
   write_name_value(I965_BATCHBUFFER_LOGGER_MESSAGE_BLOCK_BEGIN,
                    txt, fmt, va);
}

void
BatchbufferLoggerOutput::
end_block(void)
{
   if (m_block_stack.empty()) {
      return;
   }

   m_block_stack.pop_back();
   for (BatchbufferLoggerSession *s : m_sessions) {
      s->add_element(I965_BATCHBUFFER_LOGGER_MESSAGE_BLOCK_END,
                     nullptr, 0, nullptr, 0);
   }
}

void
BatchbufferLoggerOutput::
clear_block_stack(unsigned int desired_depth)
{
   while(m_block_stack.size() > desired_depth) {
      end_block();
   }
}

void
BatchbufferLoggerOutput::
print_value(const char *name, const char *fmt, ...)
{
   va_list args;
   va_start(args, fmt);
   vprint_value(name, fmt, args);
   va_end(args);
}

void
BatchbufferLoggerOutput::
vprint_value(const char *name, const char *fmt, va_list va)
{
   write_name_value(I965_BATCHBUFFER_LOGGER_MESSAGE_VALUE, name, fmt, va);
}

void
BatchbufferLoggerOutput::
write_name_value(enum i965_batchbuffer_logger_message_type_t tp,
                 const char *name, const char *fmt,
                 va_list va)
{
   if (m_sessions.empty() && tp != I965_BATCHBUFFER_LOGGER_MESSAGE_BLOCK_BEGIN) {
      return;
   }

   char buffer[4096];
   uint32_t value_length, name_length;
   va_list va_value;

   va_copy(va_value, va);
   name_length = std::strlen(name);
   value_length = std::vsnprintf(buffer, sizeof(buffer), fmt, va);

   if (value_length > sizeof(buffer)) {
      std::vector<char> tmp(value_length);
      std::vsnprintf(&tmp[0], tmp.size(), fmt, va_value);
      add_element(tp, name, name_length, &tmp[0], value_length);
   } else {
      add_element(tp, name, name_length, buffer, value_length);
   }
}

void
BatchbufferLoggerOutput::
add_element(enum i965_batchbuffer_logger_message_type_t tp,
            const void *name, uint32_t name_length,
            const void *value, uint32_t value_length)
{
   for (BatchbufferLoggerSession *s : m_sessions) {
      s->add_element(tp, name, name_length, value, value_length);
   }

   if (tp == I965_BATCHBUFFER_LOGGER_MESSAGE_BLOCK_BEGIN) {
      const uint8_t *n(static_cast<const uint8_t*>(name));
      const uint8_t *v(static_cast<const uint8_t*>(value));
      block_stack_entry b(u8vector(n, n + name_length),
                          u8vector(v, v + value_length));
      m_block_stack.push_back(b);
   }
}

/////////////////////////////////
// MessageItem methods
MessageItem::
MessageItem(uint32_t bb_location, const char *fmt, va_list ap):
   MessageActionBase(bb_location)
{
   char buffer[4096];
   unsigned int sz;

   std::printf(fmt, ap);
   sz = std::snprintf(buffer, sizeof(buffer), fmt, ap);
   if (sz > sizeof(buffer)) {
      std::vector<char> buf(sz, '\0');
      std::snprintf(&buf[0], sz, fmt, ap);
      m_text = &buf[0];
   } else {
      m_text = &buffer[0];
   }
}

MessageItem::
MessageItem(uint32_t bb_location, const char *fmt):
   MessageActionBase(bb_location),
   m_text(fmt)
{
}

void
MessageItem::
action(int block_level, BatchbufferLoggerOutput &dst) const
{
   if (dst) {
      dst.clear_block_stack(block_level);
      dst.begin_block_value("Driver Message", "@%u %s",
                            bb_location(), m_text.c_str());
   }
}

////////////////////////////////////
// APIStartCallMarker methods
void
APIStartCallMarker::
print_ioctl_log(const std::list<std::string> &ioctl_log,
                BatchbufferLoggerOutput &dst)
{
   if (dst && !ioctl_log.empty()) {
      uint32_t ioctl_message_id;
      std::list<std::string>::const_iterator iter;
      for(ioctl_message_id = 0, iter = ioctl_log.begin();
          iter != ioctl_log.end(); ++iter, ++ioctl_message_id) {
         std::ostringstream name;
         name << "IOCTL." << ioctl_message_id;
         dst.print_value(name.str().c_str(), "%s", iter->c_str());
      }
   }
}

void
APIStartCallMarker::
emit(uint32_t next_entry_start_bb_location,
     BatchbufferLoggerOutput &dst, unsigned int top_level,
     BatchbufferDecoder *decoder, const GEMBufferObject *batchbuffer) const
{
   if (dst) {
      std::ostringstream str;

      str << "Call." << m_call_id << "." << m_api_call;
      if (next_entry_start_bb_location > m_start_bb_location) {
         str << ".CreatedGPUCommands";
      }
      if (!m_ioctl_log.empty()) {
         str << ".CreateIOCTLs";
      }
      dst.clear_block_stack(top_level);
      dst.begin_block(str.str().c_str());
      dst.print_value("Call Number", "%d", m_call_id);
      dst.print_value("Function", "%s", m_api_call.c_str());
      dst.print_value("Details", "%s", m_api_call_details.c_str());
      print_ioctl_log(m_ioctl_log, dst);
   }

   uint32_t last_time(m_start_bb_location);
   const std::list<std::shared_ptr<MessageActionBase> > &items(m_items.items());
   for (auto iter = items.begin(); iter != items.end(); ++iter) {
      MessageActionBase *p;
      p = iter->get();
      if (last_time < p->bb_location() && batchbuffer) {
         if (dst) {
            dst.clear_block_stack(top_level + 2);
         }
         BatchbufferDecoder::handle_batchbuffer_contents(decoder, dst, batchbuffer,
                                                         last_time, p->bb_location());
         last_time = p->bb_location();
      }

      p->action(top_level + 1, dst);
   }

   if (batchbuffer && last_time < next_entry_start_bb_location) {
      if (dst) {
         dst.clear_block_stack(top_level + 1);
      }
      BatchbufferDecoder::handle_batchbuffer_contents(decoder, dst, batchbuffer,
                                                      last_time,
                                                      next_entry_start_bb_location);
   }
}

//////////////////////////////////
// GEMBufferObject methods
GEMBufferObject::
GEMBufferObject(int fd, const struct drm_i915_gem_create &pdata):
   m_fd(fd),
   m_handle(pdata.handle),
   m_size(pdata.size),
   m_user_ptr(nullptr),
   m_mapped(nullptr)
{
}

GEMBufferObject::
GEMBufferObject(int fd, const struct drm_i915_gem_userptr &pdata):
   m_fd(fd),
   m_handle(pdata.handle),
   m_size(pdata.user_size),
   m_user_ptr((const uint8_t*)pdata.user_ptr),
   m_mapped((void*)pdata.user_ptr)
{
   assert(m_handle != 0);
}

GEMBufferObject::
~GEMBufferObject()
{
   unmap();
}

void
GEMBufferObject::
unmap(void) const
{
   if (m_mapped && m_mapped != m_user_ptr) {
      munmap(m_mapped, m_size);
      m_mapped = nullptr;
   }
}

int
GEMBufferObject::
pread_buffer(void *dst, uint64_t start, uint64_t sz) const
{
   if (start + sz  > m_size) {
      return -1;
   }

   if (!m_user_ptr) {
      struct drm_i915_gem_pread pread_args;
      pread_args.handle = m_handle;
      pread_args.offset = start;
      pread_args.size = sz;
      pread_args.data_ptr = (__u64) dst;
      return BatchbufferLogger::local_drm_ioctl(m_fd, DRM_IOCTL_I915_GEM_PREAD, &pread_args);
   } else {
      std::memcpy(dst, m_user_ptr + start, sz);
      return 0;
   }
}

const void*
GEMBufferObject::
get_mapped(void) const
{
   if (!m_mapped) {
      struct drm_i915_gem_mmap map;
      int ret;

      std::memset(&map, 0, sizeof(map));
      map.handle = m_handle;
      map.offset = 0;
      map.size = m_size;

      ret = BatchbufferLogger::local_drm_ioctl(m_fd, DRM_IOCTL_I915_GEM_MMAP, &map);
      if (ret == 0) {
         m_mapped = (void*) map.addr_ptr;
      }
   }

   return m_mapped;
}

///////////////////////////////
// GPUCommandFieldValue methods
GPUCommandFieldValue::
GPUCommandFieldValue(const gen_field_iterator &iter):
   m_gen_type(iter.field->type.kind)
{
   /* this code is essentially taken from gen_decode.c's function
    * gen_field_iterator_next(), but rather than printing the value
    * to a string (iter.value), we extract the value to this's
    * fields.
    */
   union {
      uint64_t qw;
      float f;
   } v;

   v.qw = iter.raw_value;
   switch (iter.field->type.kind) {
   case gen_type::GEN_TYPE_INT:
      m_value.i = static_cast<int64_t>(v.qw);
      break;
   default:
   case gen_type::GEN_TYPE_UINT:
   case gen_type::GEN_TYPE_ENUM:
   case gen_type::GEN_TYPE_UNKNOWN:
      m_value.u = static_cast<uint64_t>(v.qw);
      break;
   case gen_type::GEN_TYPE_BOOL:
      m_value.b = (v.qw != 0);
      break;
   case gen_type::GEN_TYPE_FLOAT:
      m_value.f = v.f;
      break;
   case gen_type::GEN_TYPE_ADDRESS:
   case gen_type::GEN_TYPE_OFFSET:
      m_value.u = v.qw;
      break;
   case gen_type::GEN_TYPE_UFIXED:
      uint64_t uv;
      uv = static_cast<uint64_t>(v.qw);
      m_value.f = float(uv) / float(1 << iter.field->type.f);
      break;
   case gen_type::GEN_TYPE_SFIXED: {
      int64_t uv;
      uv = static_cast<int64_t>(v.qw);
      m_value.f = float(uv) / float(1 << iter.field->type.f);
      break;
   }
   }
}

template<typename T>
T
GPUCommandFieldValue::
value(void) const
{
   switch(m_gen_type) {
   case gen_type::GEN_TYPE_INT:
      return static_cast<T>(m_value.i);

   case gen_type::GEN_TYPE_BOOL:
      return static_cast<T>(m_value.b);

   case gen_type::GEN_TYPE_FLOAT:
   case gen_type::GEN_TYPE_UFIXED:
   case gen_type::GEN_TYPE_SFIXED:
      return static_cast<T>(m_value.f);

   case gen_type::GEN_TYPE_UINT:
   case gen_type::GEN_TYPE_ENUM:
   case gen_type::GEN_TYPE_UNKNOWN:
   case gen_type::GEN_TYPE_ADDRESS:
   case gen_type::GEN_TYPE_OFFSET:
   default:
      return static_cast<T>(m_value.u);
   }
}

/////////////////////////////
// GPUCommand methods
GPUCommand::
GPUCommand(void):
   m_gem_bo(nullptr),
   m_gem_bo_offset(-1),
   m_inst(nullptr),
   m_contents(nullptr),
   m_dword_length(0),
   m_command_type(gpu_command_show_value_without_gpu_state)
{}

GPUCommand::
GPUCommand(const GEMBufferObject *q, uint64_t dword_offset, struct gen_spec *spec,
           struct gen_group *grp, int override_size):
   m_gem_bo(q),
   m_gem_bo_offset(dword_offset * sizeof(uint32_t)),
   m_dword_length(0),
   m_command_type(gpu_command_show_value_without_gpu_state),
   m_pipeline_type(gpu_pipeline_gfx)
{
   complete_init(dword_offset, spec, grp, override_size);
}

GPUCommand::
GPUCommand(const GPUAddressQuery &q, struct gen_spec *spec,
           struct gen_group *grp, int override_size):
   m_gem_bo(q.m_gem_bo),
   m_gem_bo_offset(q.m_offset_into_gem_bo),
   m_dword_length(0),
   m_command_type(gpu_command_show_value_without_gpu_state),
   m_pipeline_type(gpu_pipeline_gfx)
{
   complete_init(m_gem_bo_offset / sizeof(uint32_t),
                 spec, grp, override_size);
}

void
GPUCommand::
complete_init(uint32_t dword_offset, struct gen_spec *spec,
              struct gen_group *grp, int override_size)
{
   int length;

   assert(sizeof(uint32_t) * dword_offset == m_gem_bo_offset);

   m_contents = m_gem_bo->cpu_mapped<uint32_t>() + dword_offset;
   if(spec && !grp) {
      m_inst = gen_spec_find_instruction(spec, m_contents);
   } else {
      m_inst = grp;
   }

   /* gen_group_get_length() does not actually use the parameter m_inst;
    * the length is read from extracting the GPU command type (from bits
    * [29, 31]) and then extracting the length from a set of bits (usually
    * bits [0,7]).
    */
   if (!grp || override_size == -1) {
      length = gen_group_get_length(m_inst, m_contents);
   }
   else {
      length = override_size;
   }

   if (length > 0) {
      m_dword_length = length;
   }

   if (m_inst) {
      m_command_type = get_gpu_command_type(m_inst);
      m_pipeline_type = get_gpu_pipeline_type(m_inst);
   }
}

template<typename T>
bool
GPUCommand::
extract_field_value(const char *pname, T *dst,
                    bool read_from_header) const
{
   struct gen_field_iterator iter;

   if (!inst()) {
      return false;
   }

   gen_field_iterator_init(&iter, inst(), contents_ptr(), 0, false);
   while (gen_field_iterator_next(&iter)) {
      if ((read_from_header || !gen_field_is_header(iter.field)) &&
          0 == strcmp(pname, iter.name)) {
         GPUCommandFieldValue value(iter);

         assert(!m_archived_data.empty() ||
                value.type() != gen_type::GEN_TYPE_ADDRESS);
         *dst = value.value<T>();
         return true;
      }
   }

   return false;
}

enum GPUCommand::gpu_command_type_t
GPUCommand::
get_gpu_command_type(struct gen_group *inst)
{
   uint32_t op_code;
   op_code = gen_group_get_opcode(inst);
   switch (op_code) {
   case _MI_LOAD_REGISTER_MEM: //load a register value from a GEM BO
   case _MI_LOAD_REGISTER_IMM: //load a register value from batchbuffer
   case _MI_LOAD_REGISTER_REG: //load a register value from another register
      return gpu_command_set_register;

   case STATE_BASE_ADDRESS:
      /* because STATE_BASE_ADDRESS has option to set or not set values,
       * it is not pure state and thus should be printed on encounter
       */
   case _3DSTATE_VF_INSTANCING:
      /* _3DSTATE_VF_INSTANCING sets if a named vertex attribute is
       * instanced
       */
   case _MI_NOOP:
   case _MI_BATCH_BUFFER_START:
   case _MI_BATCH_BUFFER_END:
   case _MI_STORE_REGISTER_MEM: //writes a register value to a GEM BO
   case _MI_PREDICATE:          //modify predicate value
   case _MI_ARB_CHECK:
   case _MI_ATOMIC:
   case _MI_CLFLUSH:
   case _MI_CONDITIONAL_BATCH_BUFFER_END:
   case _MI_COPY_MEM_MEM:
   case _MI_DISPLAY_FLIP:
   case _MI_FORCE_WAKEUP:
   case _MI_LOAD_SCAN_LINES_EXCL:
   case _MI_LOAD_SCAN_LINES_INCL:
   case _MI_MATH:
   case _MI_REPORT_HEAD:
   case _MI_REPORT_PERF_COUNT:
   case _MI_RS_CONTEXT:
   case _MI_RS_CONTROL:
   case _MI_RS_STORE_DATA_IMM:
   case _MI_SEMAPHORE_SIGNAL:
   case _MI_SEMAPHORE_WAIT:
   case _MI_SET_CONTEXT:
   case _MI_SET_PREDICATE:
   case _MI_STORE_DATA_IMM:
   case _MI_STORE_DATA_INDEX:
   case _MI_SUSPEND_FLUSH:
   case _MI_UPDATE_GTT:
   case _MI_USER_INTERRUPT:
   case _MI_WAIT_FOR_EVENT:
   case _3DSTATE_PIPE_CONTROL:  //3d pipeline flushing
   case MEDIA_STATE_FLUSH:      //compute/media pipeline flushing
   case _3DSTATE_PIPELINE_SELECT:
   case _3DSTATE_PIPELINE_SELECT_GM45:
      return gpu_command_show_value_without_gpu_state;

   case _3DPRIMITIVE:
   case _GPGPU_WALKER:
      return gpu_command_show_value_with_gpu_state;

   default:
      /* TODO: go through state values and correctly tag
       * what state is part of HW context and what is not.
       */
      return gpu_command_save_value_as_state_hw_context;
   }
}

enum GPUCommand::gpu_pipeline_type_t
GPUCommand::
get_gpu_pipeline_type(struct gen_group *inst)
{
   uint32_t op_code;
   op_code = gen_group_get_opcode(inst);
   switch (op_code) {
   case _GPGPU_WALKER:
   case MEDIA_INTERFACE_DESCRIPTOR_LOAD:
   case MEDIA_VFE_STATE:
   case MEDIA_CURBE_LOAD:
      return gpu_pipeline_compute;
   default:
      return gpu_pipeline_gfx;
   };
}

uint64_t
GPUCommand::
get_gpu_address(const BatchRelocs &relocs,
                uint64_t dword_offset_from_cmd_start,
                bool ignore_lower_12_bits) const
{
   const uint32_t *p;
   const GEMBufferObject *gem;
   uint64_t dword_offset_from_gem_start;

   p = contents_ptr() + dword_offset_from_cmd_start;

   /* recycle the logic/work in BatchRelocs::get_gpu_address(),
    * for reading a GPU address from memory, but set the
    * passed GEM BO and offset to a value that should never
    * be in the reloc data.
    */
   gem = (m_archived_data.empty()) ? gem_bo() : nullptr;
   dword_offset_from_gem_start = (gem) ?
      dword_offset_from_cmd_start + dword_offset() :
      ~uint64_t(0);

   return relocs.get_gpu_address(gem, dword_offset_from_gem_start,
                                 p, ignore_lower_12_bits);
}

void
GPUCommand::
archive_data(const BatchRelocs &relocs)
{
   assert(m_dword_length == 0 || m_archived_data.empty());
   if (m_dword_length > 0) {
      m_archived_data.resize(m_dword_length);
      std::copy(m_contents, m_contents + m_dword_length,
                m_archived_data.begin());
      relocs.place_relocation_values_into_buffer(m_gem_bo, m_gem_bo_offset,
                                                 &m_archived_data);
      m_contents = &m_archived_data[0];
   }
}

//////////////////////////////////////////
// i965LatchState methods
i965LatchState::
i965LatchState(void):
   m_general_state_base_address(0),
   m_surface_state_base_address(0),
   m_dynamic_state_base_address(0),
   m_instruction_base_address(0),
   m_VIEWPORT_count(-1)
{}

void
i965LatchState::
update_state(BatchbufferDecoder *decoder, BatchbufferLoggerOutput &poutput,
             const GPUCommand &cmd)
{
   GPUCommand::state_key op_code;
   const GPUCommand *p(&cmd);
   GPUCommand archived;

   if (!cmd.is_archived()) {
      archived = cmd;
      archived.archive_data(decoder->relocs());
      p = &archived;
   }

   const GPUCommand &q(*p);

   op_code = gen_group_get_opcode(q.inst());
   switch(op_code) {
   case _3DSTATE_VS:
      update_stage_values(decoder, poutput, q, &m_VS);
      break;
   case _3DSTATE_HS:
      update_stage_values(decoder, poutput, q, &m_HS);
      break;
   case _3DSTATE_DS:
      update_stage_values(decoder, poutput, q, &m_DS);
      break;
   case _3DSTATE_GS:
      update_stage_values(decoder, poutput, q, &m_GS);
      break;
   case _3DSTATE_PS:
      update_stage_values(decoder, poutput, q, &m_PS);
      break;
   case STATE_BASE_ADDRESS:
      update_state_base_address(decoder, poutput, q);
      break;
   case _3D_STATE_CLIP: {
      /* TODO: for GEN5 and before, the maximum number of
       * viewports in in _3D_STATE_GS
       */
      int v;
      if (q.extract_field_value<int>("Maximum VP Index", &v)) {
         m_VIEWPORT_count = v + 1;
      }
      break;
   }
   }
}

void
i965LatchState::
update_stage_values(BatchbufferDecoder *decoder, BatchbufferLoggerOutput &poutput,
                    const GPUCommand &q, per_stage_values *dst)
{
   int tmp;
   if (q.extract_field_value<int>("Sampler Count", &tmp)) {
      /* 3D_STATE_XS holds the number of sampler divided by 4;
       * the awful consequence is that then we only know the
       * number of sampler states to a multiple of 4.
       */
      dst->m_sampler_count = 4 * tmp;
   }

   if (q.extract_field_value<int>("Binding Table Entry Count", &tmp)) {
      dst->m_binding_table_count = tmp;
   }
}

void
i965LatchState::
update_state_base_address_helper(const GPUCommand &q,
                                 const char *value_enabled_name,
                                 uint64_t *dst, const char *value_name)
{
   bool enabled(false);
   uint64_t v;

   q.extract_field_value<bool>(value_enabled_name, &enabled);
   if (enabled && q.extract_field_value<uint64_t>(value_name, &v)) {
      *dst = v & ~uint64_t(0xFFFu);
   }
}

void
i965LatchState::
update_state_base_address(BatchbufferDecoder *decoder, BatchbufferLoggerOutput &poutput,
                          const GPUCommand &q)
{
   assert(q.is_archived());
   update_state_base_address_helper(q,
                                    "General State Base Address Modify Enable",
                                    &m_general_state_base_address,
                                    "General State Base Address");

   update_state_base_address_helper(q,
                                    "Surface State Base Address Modify Enable",
                                    &m_surface_state_base_address,
                                    "Surface State Base Address");

   update_state_base_address_helper(q,
                                    "Dynamic State Base Address Modify Enable",
                                    &m_dynamic_state_base_address,
                                    "Dynamic State Base Address");

   update_state_base_address_helper(q,
                                    "Instruction Base Address Modify Enable",
                                    &m_instruction_base_address,
                                    "Instruction Base Address");
}

///////////////////////////////////////////
// i965Registers methods
void
i965Registers::
update_state(BatchbufferDecoder *decoder, BatchbufferLoggerOutput &poutput,
             const GPUCommand &q)
{
   GPUCommand::state_key op_code;

   op_code = gen_group_get_opcode(q.inst());
   switch (op_code) {
   case _MI_LOAD_REGISTER_MEM: {
      /* An MEM register means load the register from a GEM BO.
       * We need to get the register value from the GEM BO
       * that has the value. DANGER: We are reading the value
       * from the GEM after the ioctl returns. If the GEM BO
       * was written to later in the batchbuffer, then our read
       * here will be the value it was after everything was done,
       * not when it was used.
       *
       * Should we instead record the location and offset of the
       * value instead?
       */
      uint32_t register_offset, register_value;
      uint64_t gpu_address;

      register_offset = q[1];
      gpu_address = q.get_gpu_address(decoder->relocs(), 2, false);
      register_value = decoder->ctx().pread_buffer(&register_value,
                                                   gpu_address,
                                                   sizeof(uint32_t));
      m_register_values[register_offset] = register_value;
      break;
   }

   case _MI_LOAD_REGISTER_IMM: {
      /* An IMM load has the value for the register stored directly
       * in the batchbuffer, this command can set multiple registers
       */
      for (unsigned int i = 1, endi = q.contents_size(); i < endi; i += 2) {
         uint32_t register_offset, register_value;

         register_offset = q[i];
         register_value = q[i + 1];
         m_register_values[register_offset] = register_value;
      }
      break;
   }

   case _MI_LOAD_REGISTER_REG: {
      /* command means to copy one register to another */
      uint32_t register_src_offset, register_dst_offset;
      register_src_offset = q[1];
      register_dst_offset = q[2];
      m_register_values[register_dst_offset] = m_register_values[register_src_offset];
      break;
   }
   }
}

void
i965Registers::
decode_contents(BatchbufferDecoder *decoder,
                enum GPUCommand::gpu_pipeline_type_t pipeline,
                BatchbufferLoggerOutput &poutput)
{
   /* TODO: classify registers as to what part pipeline(s)
    * they influence
    */
   (void)pipeline;

   poutput.begin_block("Register Values");
   for(const auto v : m_register_values) {
      struct gen_group *reg;
      reg = gen_spec_find_register(decoder->spec(), v.first);

      if (reg) {
         poutput.begin_block_value("Register", "%s", reg->name);
         poutput.print_value("ID", "(0x%x)", v.first);
         poutput.print_value("value", "0x%x", v.second);
      } else {
         poutput.begin_block_value("Unknown register", "(0x%x)", v.first);
         poutput.print_value("ID", "(0x%x)", v.first);
         poutput.print_value("value", "0x%x", v.second);
      }
      poutput.end_block();
   }
   poutput.end_block();
}

///////////////////////////////////////////////
// i965HWContextData methods
i965HWContextData::
i965HWContextData(uint32_t ctx_id):
   m_ctx_id(ctx_id)
{
}

i965HWContextData::
~i965HWContextData()
{
}

void
i965HWContextData::
update_state(BatchbufferDecoder *decoder, BatchbufferLoggerOutput &poutput,
             const GPUCommand &q)
{
   enum GPUCommand::gpu_command_type_t tp;
   const GPUCommand *pq(&q);

   tp = q.gpu_command_type();

   switch (tp) {
   case GPUCommand::gpu_command_save_value_as_state_hw_context: {
      uint32_t op_code;
      op_code = gen_group_get_opcode(q.inst());

      GPUCommand &dst(m_state[op_code]);
      dst = q;
      dst.archive_data(decoder->relocs());
      pq = &dst;
      break;
   }

   case GPUCommand::gpu_command_set_register: {
      /* TODO: not all registers are part of context state; some
       * are global to the entire GPU. Eventually need to adress
       * that issue.
       */
      m_registers.update_state(decoder, poutput, q);
      break;
   }

   default:
      /* TODO: should we track the values set by _3DSTATE_VF_INSTANCING? */
      break;
   }
   m_latch_state.update_state(decoder, poutput, *pq);
}

void
i965HWContextData::
decode_contents(BatchbufferDecoder *decoder,
                enum GPUCommand::gpu_pipeline_type_t pipeline,
                BatchbufferLoggerOutput &poutput)
{
   poutput.begin_block("State of Context");
   for(const auto entry : m_state) {
      if (entry.second.gpu_pipeline_type() == pipeline) {
         decoder->decode_gpu_command(poutput, entry.second);
      }
   }
   m_registers.decode_contents(decoder, pipeline, poutput);
   poutput.end_block();
}

//////////////////////////////////////
// GPUState methods
void
GPUState::
update_state(BatchbufferDecoder *decoder, BatchbufferLoggerOutput &poutput,
             const GPUCommand &q)
{
   if (q.gpu_command_type() ==
       GPUCommand::gpu_command_save_value_as_state_not_hw_context) {
      GPUCommand::state_key op_code;
      op_code = gen_group_get_opcode(q.inst());

      GPUCommand &dst(m_state[op_code]);
      dst = q;
      dst.archive_data(decoder->relocs());
   } else {
      m_ctx_data->update_state(decoder, poutput, q);
   }
}

void
GPUState::
decode_contents(BatchbufferDecoder *decoder,
                enum GPUCommand::gpu_pipeline_type_t pipeline,
                BatchbufferLoggerOutput &poutput)
{
   m_ctx_data->decode_contents(decoder, pipeline, poutput);
   if (!m_state.empty()) {
      poutput.begin_block("State of GPU, not of Context");
      for(const auto entry : m_state) {
         if (entry.second.gpu_pipeline_type() == pipeline) {
            decoder->decode_gpu_command(poutput, entry.second);
         }
      }
      poutput.end_block();
   }
}

///////////////////////////////////////////////
// BatchbufferDecoder::DetailedDecoder methods
BatchbufferDecoder::DetailedDecoder::
DetailedDecoder(void)
{
   m_elements[MEDIA_INTERFACE_DESCRIPTOR_LOAD] =
      &BatchbufferDecoder::decode_media_interface_descriptor_load;
   m_elements[_3DSTATE_VS] = &BatchbufferDecoder::decode_3dstate_xs;
   m_elements[_3DSTATE_GS] = &BatchbufferDecoder::decode_3dstate_xs;
   m_elements[_3DSTATE_DS] = &BatchbufferDecoder::decode_3dstate_xs;
   m_elements[_3DSTATE_HS] = &BatchbufferDecoder::decode_3dstate_xs;
   m_elements[_3DSTATE_PS] = &BatchbufferDecoder::decode_3dstate_ps;

   m_elements[_3DSTATE_BINDING_TABLE_POINTERS_VS] =
      &BatchbufferDecoder::decode_3dstate_binding_table_pointers_vs;
   m_elements[_3DSTATE_BINDING_TABLE_POINTERS_HS] =
      &BatchbufferDecoder::decode_3dstate_binding_table_pointers_hs;
   m_elements[_3DSTATE_BINDING_TABLE_POINTERS_DS] =
      &BatchbufferDecoder::decode_3dstate_binding_table_pointers_ds;
   m_elements[_3DSTATE_BINDING_TABLE_POINTERS_GS] =
      &BatchbufferDecoder::decode_3dstate_binding_table_pointers_gs;
   m_elements[_3DSTATE_BINDING_TABLE_POINTERS_PS] =
      &BatchbufferDecoder::decode_3dstate_binding_table_pointers_ps;

   m_elements[_3DSTATE_SAMPLER_STATE_POINTERS_VS] =
      &BatchbufferDecoder::decode_3dstate_sampler_state_pointers_vs;
   m_elements[_3DSTATE_SAMPLER_STATE_POINTERS_DS] =
      &BatchbufferDecoder::decode_3dstate_sampler_state_pointers_hs;
   m_elements[_3DSTATE_SAMPLER_STATE_POINTERS_HS] =
      &BatchbufferDecoder::decode_3dstate_sampler_state_pointers_ds;
   m_elements[_3DSTATE_SAMPLER_STATE_POINTERS_GS] =
      &BatchbufferDecoder::decode_3dstate_sampler_state_pointers_gs;
   m_elements[_3DSTATE_SAMPLER_STATE_POINTERS_PS] =
      &BatchbufferDecoder::decode_3dstate_sampler_state_pointers_ps;
   m_elements[_3DSTATE_SAMPLER_STATE_POINTERS] =
      &BatchbufferDecoder::decode_3dstate_sampler_state_pointers_gen6;

   m_elements[_3DSTATE_VIEWPORT_STATE_POINTERS_CC] =
      &BatchbufferDecoder::decode_3dstate_viewport_state_pointers_cc;
   m_elements[_3DSTATE_VIEWPORT_STATE_POINTERS_SF_CLIP] =
      &BatchbufferDecoder::decode_3dstate_viewport_state_pointers_sf_clip;
   m_elements[_3DSTATE_BLEND_STATE_POINTERS] =
      &BatchbufferDecoder::decode_3dstate_blend_state_pointers;
   m_elements[_3DSTATE_CC_STATE_POINTERS] =
      &BatchbufferDecoder::decode_3dstate_cc_state_pointers;
   m_elements[_3DSTATE_SCISSOR_STATE_POINTERS] =
      &BatchbufferDecoder::decode_3dstate_scissor_state_pointers;
}

void
BatchbufferDecoder::DetailedDecoder::
decode(BatchbufferDecoder *decoder, BatchbufferLoggerOutput &poutput,
       const GPUCommand &data)
{
   static DetailedDecoder R;
   std::map<uint32_t, fcn>::const_iterator iter;
   uint32_t opcode;

   opcode = gen_group_get_opcode(data.inst());
   iter = R.m_elements.find(opcode);
   if (iter != R.m_elements.end()) {
      fcn function(iter->second);
      (decoder->*function)(poutput, data);
   }
}

//////////////////////////////////////////////
// BatchRelocs methods
void
BatchRelocs::
add_entries(const struct drm_i915_gem_exec_object2 &exec_object,
            GEMBufferObject *q, GEMBufferTracker *tracker,
            PerHWContext *ctx,
            const struct drm_i915_gem_exec_object2 *lut_references)
{
   struct drm_i915_gem_relocation_entry *reloc_entries;

   reloc_entries = (struct drm_i915_gem_relocation_entry*) exec_object.relocs_ptr;

   for (unsigned int r = 0; r < exec_object.relocation_count; ++r) {
      uint32_t gem_bo_handle;
      GEMBufferObject *bo;
      uint64_t gpu_address;

      gem_bo_handle = reloc_entries[r].target_handle;
      if (lut_references) {
         gem_bo_handle = lut_references[gem_bo_handle].handle;
      }

      bo = tracker->fetch_gem_bo(gem_bo_handle);
      if (!bo) {
         continue;
      }

      gpu_address = bo->gpu_address_begin(ctx->ctx_id()) + reloc_entries[r].delta;

      /* When reading an address from BO q at offset, we will
       * read the gpu_address below.
       */
      add_entry(q, reloc_entries[r].offset, gpu_address);
   }
}

void
BatchRelocs::
emit_reloc_data(BatchbufferLoggerOutput &poutput)
{
   poutput.begin_block("Relocs");
   for(const auto &v : m_relocs) {

      if(v.second.empty()) {
         continue;
      }

      poutput.begin_block_value("Relocs on GEM", "%u", v.first->handle());
      for(const auto &w : v.second) {
         poutput.begin_block("Reloc Entry");
         poutput.print_value("Offset", "0x%012" PRIx64, w.first);
         poutput.print_value("GPU Address", "0x%012" PRIx64, w.second);
         poutput.end_block();
      }
      poutput.end_block();
   }
   poutput.end_block();
}

void
BatchRelocs::
place_relocation_values_into_buffer(const GEMBufferObject *gem, uint64_t gem_bo_offset,
                                    std::vector<uint32_t> *dst) const
{
   reloc_map::const_iterator gem_iter;
   reloc_map_of_gem_bo::const_iterator reloc_iter;
   unsigned int dst_end;

   gem_iter = m_relocs.find(gem);

   if (gem_iter == m_relocs.end()) {
      return;
   }

   dst_end = sizeof(uint32_t) * dst->size() + gem_bo_offset;

   for(reloc_iter = gem_iter->second.lower_bound(gem_bo_offset);
       reloc_iter != gem_iter->second.end() && reloc_iter->first < dst_end;
       ++reloc_iter)
   {
      unsigned int s;
      uint64_t addr;

      addr = reloc_iter->second;

      assert(reloc_iter->first >= gem_bo_offset);
      s = reloc_iter->first - gem_bo_offset;

      /* Recall that the locations in BatchRelocs are copied
       * directly from the kernel and are in units of bytes,
       * NOT DWORD's.
       */
      assert(s % sizeof(uint32_t) == 0);
      s /= sizeof(uint32_t);
      assert(s < dst->size());

      (*dst)[s] = addr & 0xFFFFFFFF;
      if (!m_32bit_gpu_addresses) {
         assert(s + 1 < dst->size());
         /* preserve the high 16 bits since the address
          * is 48-bits wide and there may be additional
          * data stashed in those highest 16-bits.
          */
         (*dst)[s + 1] &= 0xFFFF0000u;
         (*dst)[s + 1] |= (addr >> 32u) & 0x0000FFFFu;
      }
   }
}

uint64_t
BatchRelocs::
get_gpu_address(const GEMBufferObject *q, uint64_t dword_offset,
                const uint32_t *p, bool ignore_lower_12_bits) const
{
   reloc_map::const_iterator gem_iter;

   uint64_t addr = p[0];
   if (!m_32bit_gpu_addresses) {
      /* On BDW and above, the address is 48-bits wide, consuming
       * an aditional _32_ bits. Grab the next 16-bits of the
       * address
       */
      addr |= uint64_t(p[1] & 0xFFFF) << uint64_t(32);
   }

   gem_iter = m_relocs.find(q);
   if (gem_iter != m_relocs.end()) {
      reloc_map_of_gem_bo::const_iterator reloc_iter;
      reloc_iter = gem_iter->second.find(sizeof(uint32_t) * dword_offset);
      if (reloc_iter != gem_iter->second.end()) {
         addr = reloc_iter->second;
      }
   }

   /* Address are to be page aligned (i.e. last 12 bits are zero),
    * but HW commands might stash extra data in those 12-bits,
    * zero those bits out.
    */
   return ignore_lower_12_bits ?
      addr & ~uint64_t(0xFFFu) :
      addr;
}

///////////////////////////////////////////////
// BatchbufferDecoder methods
BatchbufferDecoder::
BatchbufferDecoder(enum decode_level_t decode_level,
                   enum print_reloc_level_t print_reloc_level,
                   bool decode_shaders, bool organize_by_ioctls,
                   struct gen_spec *spec, struct gen_disasm *dis,
                   int pciid, GEMBufferTracker *tracker,
                   bool is_post_call, BatchRelocs &relocs,
                   GPUCommandCounter *gpu_command_counter,
                   ShaderFileList *shader_filelist,
                   struct drm_i915_gem_execbuffer2 *execbuffer2):
   m_decode_level(decode_level),
   m_print_reloc_level(print_reloc_level),
   m_decode_shaders(decode_shaders),
   m_organize_by_ioctls(organize_by_ioctls),
   m_spec(spec),
   m_gen_disasm(dis),
   m_pci_id(pciid),
   m_tracker(tracker),
   m_gpu_command_counter(gpu_command_counter),
   m_shader_filelist(shader_filelist),
   m_buffers(execbuffer2->buffer_count),
   m_reloc_handles_are_indices(execbuffer2->flags & I915_EXEC_HANDLE_LUT),
   m_ctx(m_tracker->fetch_hw_context(execbuffer2->rsvd1)),
   m_gpu_state(m_ctx),
   m_relocs(relocs),
   m_execbuffer2(execbuffer2)
{
   struct drm_i915_gem_exec_object2 *exec_objects;
   const struct drm_i915_gem_exec_object2 *lut_references;

   exec_objects = (struct drm_i915_gem_exec_object2 *) (uintptr_t) execbuffer2->buffers_ptr;
   for(unsigned int i = 0; i < execbuffer2->buffer_count; ++i) {
      m_buffers[i] = m_tracker->fetch_gem_bo(exec_objects[i].handle);
   }

   if (execbuffer2->flags & I915_EXEC_BATCH_FIRST) {
      m_batchbuffer = m_buffers.front();
   } else {
      m_batchbuffer = m_buffers.back();
   }

   assert(m_tracker);
   assert(m_ctx);

   m_batchbuffer_log = m_tracker->fetch_or_create(nullptr, m_batchbuffer->handle());
   assert(m_batchbuffer_log);

   if (execbuffer2->flags & I915_EXEC_HANDLE_LUT) {
      lut_references = exec_objects;
   } else {
      lut_references = nullptr;
   }

   /* Update GEM BO's and Reloc information only if the decoding
    * happens AFTER the ioctl. If the decoding happens before the
    * ioctl, then the values stored in exec_objects[] are the guesses
    * from the driver, not necessarily the correct values.
    */
   if (is_post_call) {
      /* first update the GEM BO GPU Addresses */
      for(unsigned int i = 0; i < execbuffer2->buffer_count; ++i) {
         tracker->update_gem_bo_gpu_address(m_ctx, &exec_objects[i]);
      }

      for(unsigned int i = 0; i < execbuffer2->buffer_count; ++i) {
         /* Bah humbug; the kernel interface does not state that
          * the address values in a batchbuffer will get updated;
          * The upshot is that we then need to examine the reloc
          * data of the ioctl call.
          */
         GEMBufferObject *q;
         q = tracker->fetch_gem_bo(exec_objects[i].handle);
         if (!q) {
            continue;
         }
         m_relocs.add_entries(exec_objects[i], q, tracker, m_ctx, lut_references);
      }
   }
}

BatchbufferDecoder::
~BatchbufferDecoder()
{
   for(GEMBufferObject *bo : m_buffers) {
      bo->unmap();
   }
}

void
BatchbufferDecoder::
decode_shader(BatchbufferLoggerOutput &poutput, enum shader_decode_entry_t tp,
              uint64_t gpu_address)
{
   const void *shader;
   GPUAddressQuery query;
   static const char *labels[shader_decode_entry_count] = {
      [shader_decode_vs] = "VertexShader",
      [shader_decode_hs] = "HullShader",
      [shader_decode_ds] = "DomainShader",
      [shader_decode_gs] = "GeometryShader",
      [shader_decode_ps_8] = "PS8-Shader",
      [shader_decode_ps_16] = "PS16-Shader",
      [shader_decode_ps_32] = "PS32-Shader",
      [shader_decode_media_compute] = "MediaComputeShader",
   };

   poutput.begin_block(labels[tp]);

   shader = m_ctx->cpu_mapped<void>(gpu_address, &query);
   poutput.print_value("GPU Address", "0x%012" PRIx64, gpu_address);
   if (shader && query.m_gem_bo) {
      if (m_decode_shaders) {
         const char *disasm;
         disasm = m_shader_filelist->disassembly(shader, m_pci_id, m_gen_disasm);
         if (disasm) {
            poutput.print_value("Assembly", "%s", disasm);
         }
      } else {
         const char *filename;
         filename = m_shader_filelist->filename(shader, m_pci_id, m_gen_disasm, labels[tp]);
         if (filename) {
            poutput.print_value("ShaderFile", "%s", filename);
         }
      }
   } else {
      poutput.print_value("GPU Address", "0x%012" PRIx64 "(BAD)", gpu_address);
   }

   poutput.end_block();
}

void
BatchbufferDecoder::
emit_execbuffer2_details(BatchbufferLoggerOutput &poutput)
{
   poutput.begin_block("drmIoctl(execbuffer2) details");
   poutput.print_value("fd", "%d", m_batchbuffer_log->src()->fd);
   poutput.print_value("Context", "%u", m_execbuffer2->rsvd1);
   poutput.print_value("GEM BO", "%u", m_batchbuffer_log->src()->gem_bo);
   poutput.print_value("bytes", "%d", m_execbuffer2->batch_len);
   poutput.print_value("dwords", "%d", m_execbuffer2->batch_len / 4);
   poutput.print_value("start", "%d", m_execbuffer2->batch_start_offset);
   poutput.begin_block("exec-objects");
   for (unsigned int i = 0; i < m_buffers.size(); ++i) {
      poutput.begin_block_value("buffer", "%u", i);
      poutput.print_value("GEM BO", "%u", m_buffers[i]->handle());
      poutput.print_value("ptr", "%p", m_buffers[i]);
      poutput.print_value("GPU Address", "0x%012" PRIx64,
                          m_buffers[i]->gpu_address_begin(m_ctx->ctx_id()));
      poutput.end_block();
   }
   poutput.end_block(); //exec-objects
   if (m_print_reloc_level >= print_reloc_gem_gpu_updates) {
      m_relocs.emit_reloc_data(poutput);
   }
   poutput.end_block(); //drmIoctl(execbuffer2) details
}

void
BatchbufferDecoder::
emit_log(BatchbufferLoggerOutput &poutput, int count)
{
   assert(m_batchbuffer_log);
   m_gpu_command_counter->add_batch();

   if (poutput) {
      const char *blockname;
      blockname = (m_organize_by_ioctls) ?
         "(execbuffer2) IOCTL":
         "---- Batchbuffer Begin ----";

      if (!m_batchbuffer_log->empty()) {
         poutput.begin_block_value(blockname, "#%d, CallIds=[%u, %u]",
                                   count, m_batchbuffer_log->first_api_call_id(),
                                   m_batchbuffer_log->last_api_call_id());
      } else {
         poutput.begin_block_value(blockname, "#%d", count);
      }

      if (m_organize_by_ioctls) {
         emit_execbuffer2_details(poutput);
         poutput.begin_block("Contents");
      }
   }

   if (m_ctx->ctx_id() == 0) {
      m_ctx->reset_state();
   }

   m_batchbuffer_log->emit_log(this, poutput, m_batchbuffer,
                               m_execbuffer2->batch_start_offset / 4,
                               m_execbuffer2->batch_len / 4,
                               count);

   if (poutput) {
      if (!m_organize_by_ioctls) {
         emit_execbuffer2_details(poutput);
      } else {
         poutput.end_block(); //Contents
      }
      poutput.end_block(); //Batchbuffer Begin
   }
}


void
BatchbufferDecoder::
decode_media_interface_descriptor_load(BatchbufferLoggerOutput &poutput, const GPUCommand &data)
{
   uint64_t gpu_address, descriptor_start;
   int length(0);
   struct gen_group *grp;

   grp = gen_spec_find_struct(m_spec, "INTERFACE_DESCRIPTOR_DATA");
   if (!grp) {
      return;
   }

   if (data.extract_field_value<uint64_t>("Interface Descriptor Data Start Address",
                                          &descriptor_start)) {
      gpu_address = m_gpu_state.ctx().m_latch_state.m_dynamic_state_base_address + descriptor_start;
   } else {
      poutput.print_value("Unable to extract field \"Interface Descriptor Data Start Address\"", "");
      std::fprintf(stderr, "!");
      return;
   }

   data.extract_field_value<int>("Interface Descriptor Total Length", &length);
   length /= 32;

   for(int i = 0; i < length; ++i, gpu_address += 8 * sizeof(uint32_t)) {
      GPUAddressQuery address_query(m_ctx->get_memory_at_gpu_address(gpu_address));
      /* we need to override the length value; this is because the function
       * gen_group_get_length() does not read the length value from the XML ever,
       * instead it reads the first uint32_t and from that get the length; however
       * that is only correct for commands to be placed in a batchbuffer, not
       * for structs.
       */
      GPUCommand descriptor(address_query, m_spec, grp, 8);
      uint64_t shader_offset, shader_gpu_address;
      int tmp, binding_table_count, sampler_count;
      uint32_t sampler_pointer, binding_table_pointer;

      poutput.begin_block_value("Descriptor", "#%d", i);
      poutput.print_value("type", "%s", descriptor.inst()->name);
      poutput.print_value("GPU Address", "%012" PRIx64, gpu_address);
      decode_gen_group(poutput, descriptor.gem_bo(), descriptor.dword_offset(),
                       descriptor.contents_ptr(), descriptor.inst());

      /* the correct thing to do would be to call descriptor.extract_field_value()
       * passing as the name "Kernel Start Pointer", to get shader_offset; however,
       * for some reason, gen_decoder does not see that field. Indeed iterating over
       * the fields in grp by hand we see that there is no "Kernel Start Pointer" field
       * in the list of fields;
       */
      shader_offset = descriptor[0];
      shader_gpu_address = m_gpu_state.ctx().m_latch_state.m_instruction_base_address + shader_offset;
      decode_shader(poutput, shader_decode_media_compute, shader_gpu_address);

      sampler_count = -1;
      if (descriptor.extract_field_value<int>("Sampler Count", &tmp)) {
         sampler_count = 4 * tmp;
      }

      binding_table_count = -1;
      if (descriptor.extract_field_value<int>("Binding Table Entry Count", &tmp)) {
         binding_table_count = tmp;
      }

      if (descriptor.extract_field_value<uint32_t>("Sampler State Pointer", &sampler_pointer)) {
         decode_3dstate_sampler_state_pointers_helper(poutput, sampler_pointer, sampler_count);
      } else {
         poutput.print_value("Unable to extract \"Sampler State Pointer\"", "");
      }

      if (descriptor.extract_field_value<uint32_t>("Binding Table Pointer", &binding_table_pointer)) {
         decode_3dstate_binding_table_pointers(poutput, "MEDIA", binding_table_pointer, binding_table_count);
      } else {
         poutput.print_value("Unable to extract \"Binding Table Pointer\"", "");
      }

      poutput.end_block();
   }
}

void
BatchbufferDecoder::
decode_3dstate_xs(BatchbufferLoggerOutput &poutput, const GPUCommand &data)
{
   bool has_shader(false);
   uint64_t offset(0), gpu_address;
   uint32_t opcode;
   enum shader_decode_entry_t shader_tp;

   data.extract_field_value<bool>("Enable", &has_shader);
   has_shader = has_shader
      && data.extract_field_value<uint64_t>("Kernel Start Pointer", &offset);

   if(!has_shader) {
      return;
   }

   opcode = gen_group_get_opcode(data.inst());
   switch(opcode) {
   default:
   case _3DSTATE_VS:
      shader_tp = shader_decode_vs;
      break;
   case _3DSTATE_HS:
      shader_tp = shader_decode_hs;
      break;
   case _3DSTATE_DS:
      shader_tp = shader_decode_ds;
      break;
   case _3DSTATE_GS:
      shader_tp = shader_decode_gs;
      break;
   }

   gpu_address = m_gpu_state.ctx().m_latch_state.m_instruction_base_address + offset;
   decode_shader(poutput, shader_tp, gpu_address);
}

void
BatchbufferDecoder::
decode_3dstate_ps(BatchbufferLoggerOutput &poutput, const GPUCommand &data)
{
   typedef std::pair<enum shader_decode_entry_t, const char*> decode_job;
   std::vector<decode_job> decode_jobs;
   bool has_8(false), has_16(false), has_32(false);
   int numEnabled;
   static const char *kernels[3] = {
      "Kernel Start Pointer 0",
      "Kernel Start Pointer 1",
      "Kernel Start Pointer 2",
   };

   data.extract_field_value<bool>("8 Pixel Dispatch Enable", &has_8);
   data.extract_field_value<bool>("16 Pixel Dispatch Enable", &has_16);
   data.extract_field_value<bool>("32 Pixel Dispatch Enable", &has_32);

   /* GEN is amusing at times, depending on what dispatches are enabled,
    * which kernel is used for different dispatch modes changes.
    *
    * | 8-enabled | 16-enabled | 32-enabled | 8-shader | 16-shader | 32-shader |
    * |  TRUE     |  FALSE     |  FALSE     | Kerenl0  |           |           |
    * |  TRUE     |  TRUE      |  FALSE     | Kerenl0  | Kerenl2   |           |
    * |  TRUE     |  TRUE      |  TRUE      | Kernel0  | Kerenl2   | Kernel1   |
    * |  FALSE    |  TRUE      |  FALSE     |          | Kernal0   |           |
    * |  FALSE    |  FALSE     |  TRUE      |          |           | Kernel0   |
    * |  FALSE    |  TRUE      |  TRUE      |          | Kernel2   | Kernel1   |
    *
    * Atleast from the table, we can get a simple set or rules:
    *  - 8-wide, if it is enabled, it is alway at Kernel0
    *  - if N-wide is the only one enabled, then it is at Kernel0
    *  - if there are atleast 2-enables, then 16-wide is at 2 and 32-wide is at 1.
    */
   numEnabled = int(has_8) + int(has_16) + int(has_32);
   if (has_8) {
      decode_jobs.push_back(decode_job(shader_decode_ps_8, kernels[0]));
   }

   if (numEnabled > 1) {
      if (has_16) {
         decode_jobs.push_back(decode_job(shader_decode_ps_16, kernels[2]));
      }
      if (has_32) {
         decode_jobs.push_back(decode_job(shader_decode_ps_32, kernels[1]));
      }
   } else {
      if (has_16) {
         decode_jobs.push_back(decode_job(shader_decode_ps_16, kernels[0]));
      }
      else if (has_32) {
         decode_jobs.push_back(decode_job(shader_decode_ps_32, kernels[0]));
      }
   }

   for (const decode_job &J : decode_jobs) {
      uint64_t addr;
      if (data.extract_field_value<uint64_t>(J.second, &addr)) {
         addr += m_gpu_state.ctx().m_latch_state.m_instruction_base_address;
         decode_shader(poutput, J.first, addr);
      }
   }
}

void
BatchbufferDecoder::
decode_3dstate_constant(BatchbufferLoggerOutput &poutput,
                               const GPUCommand &data)
{
}

void
BatchbufferDecoder::
decode_3dstate_binding_table_pointers_vs(BatchbufferLoggerOutput &poutput,
                                         const GPUCommand &data)
{
   decode_3dstate_binding_table_pointers(poutput, "VS", data[1] & ~0x1fu,
                                         m_gpu_state.ctx().m_latch_state.m_VS.m_binding_table_count);
}

void
BatchbufferDecoder::
decode_3dstate_binding_table_pointers_ds(BatchbufferLoggerOutput &poutput,
                                         const GPUCommand &data)
{
   decode_3dstate_binding_table_pointers(poutput, "DS", data[1] & ~0x1fu,
                                         m_gpu_state.ctx().m_latch_state.m_DS.m_binding_table_count);
}

void
BatchbufferDecoder::
decode_3dstate_binding_table_pointers_hs(BatchbufferLoggerOutput &poutput,
                                         const GPUCommand &data)
{
   decode_3dstate_binding_table_pointers(poutput, "HS", data[1] & ~0x1fu,
                                         m_gpu_state.ctx().m_latch_state.m_HS.m_binding_table_count);
}

void
BatchbufferDecoder::
decode_3dstate_binding_table_pointers_ps(BatchbufferLoggerOutput &poutput,
                                                const GPUCommand &data)
{
   decode_3dstate_binding_table_pointers(poutput, "PS", data[1] & ~0x1fu,
                                         m_gpu_state.ctx().m_latch_state.m_PS.m_binding_table_count);
}

void
BatchbufferDecoder::
decode_3dstate_binding_table_pointers_gs(BatchbufferLoggerOutput &poutput,
                                         const GPUCommand &data)
{
   decode_3dstate_binding_table_pointers(poutput, "GS", data[1] & ~0x1fu,
                                         m_gpu_state.ctx().m_latch_state.m_GS.m_binding_table_count);
}

void
BatchbufferDecoder::
decode_3dstate_binding_table_pointers(BatchbufferLoggerOutput &poutput,
                                      const std::string &label, uint32_t offset,
                                      int cnt)
{
   struct gen_group *surface_state;
   uint64_t gpu_address;
   GPUAddressQuery Q;
   const uint32_t *v;

   /* The command is essentially just provides an address (given as an
    * offset from surface_state_base_address) of a sequence of values V.
    * That sequence of values V is just a sequence of offsets also from
    * surface_state_base_address which is the location of the surface
    * state values.
    */
   surface_state = gen_spec_find_struct(m_spec, "RENDER_SURFACE_STATE");
   gpu_address = offset + m_gpu_state.ctx().m_latch_state.m_surface_state_base_address;
   v = m_ctx->cpu_mapped<uint32_t>(gpu_address, &Q);

   if (!Q.m_gem_bo || !surface_state) {
      return;
   }

   poutput.begin_block_value("Binding Tables", "%s", label.c_str());

   /* i965 driver does "Track-ish" the number of binding table entries in
    * each program stage, the value of X.base.binding_table.size_bytes /4
    * is the number of entries for a stage X where X is brw->wm,
    * brw->vs, brw->gs, brw->tcs and brw->tes
    */
   if (cnt < 0) {
      cnt = 16;
      poutput.print_value("Count", "%d (Guessing)", cnt);
   } else {
      poutput.print_value("Count", "%d", cnt);
   }

   for (int i = 0; i < cnt; ++i) {
      uint64_t state_gpu_address;
      const uint32_t *state_ptr;
      GPUAddressQuery SQ;

      if (v[i] == 0) {
         continue;
      }

      poutput.begin_block_value("Binding Table", "#%d", i);
      poutput.print_value("offset", "%u", v[i]);

      state_gpu_address = v[i] + m_gpu_state.ctx().m_latch_state.m_surface_state_base_address;
      state_ptr = m_ctx->cpu_mapped<uint32_t>(state_gpu_address, &SQ);
      if (!SQ.m_gem_bo) {
         poutput.print_value("GPU address", "0x%012" PRIx64 " (BAD)", state_gpu_address);
         poutput.end_block();
         continue;
      }

      poutput.print_value("GPU address", "0x%012" PRIx64, state_gpu_address);
      decode_gen_group(poutput, SQ.m_gem_bo, SQ.m_offset_into_gem_bo, state_ptr, surface_state);

      poutput.end_block();
   }

   poutput.end_block();
}

void
BatchbufferDecoder::
decode_3dstate_sampler_state_pointers_vs(BatchbufferLoggerOutput &poutput,
                                         const GPUCommand &data)
{
   int cnt;
   cnt = m_gpu_state.ctx().m_latch_state.m_VS.m_sampler_count;
   decode_3dstate_sampler_state_pointers_helper(poutput, data[1], cnt);
}

void
BatchbufferDecoder::
decode_3dstate_sampler_state_pointers_gs(BatchbufferLoggerOutput &poutput,
                                         const GPUCommand &data)
{
   int cnt;
   cnt = m_gpu_state.ctx().m_latch_state.m_GS.m_sampler_count;
   decode_3dstate_sampler_state_pointers_helper(poutput, data[1], cnt);
}

void
BatchbufferDecoder::
decode_3dstate_sampler_state_pointers_hs(BatchbufferLoggerOutput &poutput,
                                         const GPUCommand &data)
{
   int cnt;
   cnt = m_gpu_state.ctx().m_latch_state.m_HS.m_sampler_count;
   decode_3dstate_sampler_state_pointers_helper(poutput, data[1], cnt);
}

void
BatchbufferDecoder::
decode_3dstate_sampler_state_pointers_ds(BatchbufferLoggerOutput &poutput,
                                         const GPUCommand &data)
{
   int cnt;
   cnt = m_gpu_state.ctx().m_latch_state.m_DS.m_sampler_count;
   decode_3dstate_sampler_state_pointers_helper(poutput, data[1], cnt);
}

void
BatchbufferDecoder::
decode_3dstate_sampler_state_pointers_ps(BatchbufferLoggerOutput &poutput,
                                         const GPUCommand &data)
{
   int cnt;
   cnt = m_gpu_state.ctx().m_latch_state.m_PS.m_sampler_count;
   decode_3dstate_sampler_state_pointers_helper(poutput, data[1], cnt);
}

void
BatchbufferDecoder::
decode_3dstate_sampler_state_pointers_gen6(BatchbufferLoggerOutput &poutput,
                                           const GPUCommand &data)
{
   int sample_counts[3] = {
      m_gpu_state.ctx().m_latch_state.m_VS.m_sampler_count,
      m_gpu_state.ctx().m_latch_state.m_GS.m_sampler_count,
      m_gpu_state.ctx().m_latch_state.m_PS.m_sampler_count
   };

   for (unsigned int stage = 0; stage < 3; ++stage) {
      int cnt;
      cnt = sample_counts[stage];
      decode_3dstate_sampler_state_pointers_helper(poutput, data[stage + 1], cnt);
   }
}

void
BatchbufferDecoder::
decode_3dstate_sampler_state_pointers_helper(BatchbufferLoggerOutput &poutput,
                                             uint32_t offset, int cnt)
{
   struct gen_group *g;
   uint64_t gpu_address;

   g = gen_spec_find_struct(m_spec, "SAMPLER_STATE");
   poutput.begin_block("SAMPLER_STATEs");

   if (cnt < 0) {
      cnt = 4;
      poutput.print_value("Count", "%d (Guessing)", cnt);
   } else {
      poutput.print_value("Count", "%d", cnt);
   }

   gpu_address = m_gpu_state.ctx().m_latch_state.m_dynamic_state_base_address + offset;
   for (int i = 0; i < cnt; ++i) {
      poutput.begin_block_value("SamplerState", "#%d", i);
      decode_pointer_helper(poutput, g, gpu_address);
      poutput.end_block();
   }

   poutput.end_block();
}

void
BatchbufferDecoder::
decode_3dstate_viewport_state_pointers_cc(BatchbufferLoggerOutput &poutput,
                                          const GPUCommand &data)
{
   uint64_t gpu_address;
   struct gen_group *g;

   g = gen_spec_find_struct(m_spec, "CC_VIEWPORT");
   gpu_address = m_gpu_state.ctx().m_latch_state.m_dynamic_state_base_address + (data[1] & ~0x1fu);

   poutput.begin_block("CC_VIEWPORTs");

   uint32_t cnt;
   if (m_gpu_state.ctx().m_latch_state.m_VIEWPORT_count < 0) {
      cnt = 4;
      poutput.print_value("Count", "%d (Guessing)", cnt);
   } else {
      cnt = m_gpu_state.ctx().m_latch_state.m_VIEWPORT_count;
      poutput.print_value("Count", "%d", cnt);
   }

   for (uint32_t i = 0; i < cnt; ++i) {
      poutput.begin_block_value("CC-Viewport", "#%d", i);
      decode_pointer_helper(poutput, g, gpu_address + i * 8);
      poutput.end_block();
   }

   poutput.end_block();
}

void
BatchbufferDecoder::
decode_3dstate_viewport_state_pointers_sf_clip(BatchbufferLoggerOutput &poutput,
                                               const GPUCommand &data)
{
   uint64_t gpu_address;
   struct gen_group *g;

   g = gen_spec_find_struct(m_spec, "SF_CLIP_VIEWPORT");
   gpu_address = m_gpu_state.ctx().m_latch_state.m_dynamic_state_base_address + (data[1] & ~0x3fu);

   poutput.begin_block("SF_CLIP_VIEWPORTs");

   uint32_t cnt;
   if (m_gpu_state.ctx().m_latch_state.m_VIEWPORT_count < 0) {
      cnt = 4;
      poutput.print_value("Count", "%d (Guessing)", cnt);
   } else {
      cnt = m_gpu_state.ctx().m_latch_state.m_VIEWPORT_count;
      poutput.print_value("Count", "%d", cnt);
   }

   for (uint32_t i = 0; i < cnt; ++i) {
      poutput.begin_block_value("Viewport", "#%d", i);
      decode_pointer_helper(poutput, g, gpu_address + i * 64);
      poutput.end_block();
   }

   poutput.end_block();
}

void
BatchbufferDecoder::
decode_3dstate_blend_state_pointers(BatchbufferLoggerOutput &poutput,
                                    const GPUCommand &data)
{
   uint64_t gpu_address;

   gpu_address = m_gpu_state.ctx().m_latch_state.m_dynamic_state_base_address + (data[1] & ~0x3fu);
   poutput.begin_block("BLEND_STATE");
   decode_pointer_helper(poutput, "BLEND_STATE", gpu_address);
   poutput.end_block();
}

void
BatchbufferDecoder::
decode_3dstate_cc_state_pointers(BatchbufferLoggerOutput &poutput,
                                 const GPUCommand &data)
{
   uint64_t gpu_address;

   gpu_address = m_gpu_state.ctx().m_latch_state.m_dynamic_state_base_address + (data[1] & ~0x3fu);
   poutput.begin_block("COLOR_CALC_STATE");
   decode_pointer_helper(poutput, "COLOR_CALC_STATE", gpu_address);
   poutput.end_block();
}

void
BatchbufferDecoder::
decode_3dstate_scissor_state_pointers(BatchbufferLoggerOutput &poutput,
                                      const GPUCommand &data)
{
   uint64_t gpu_address;

   gpu_address = m_gpu_state.ctx().m_latch_state.m_dynamic_state_base_address + (data[1] & ~0x1fu);
   poutput.begin_block("SCISSOR_RECT");
   decode_pointer_helper(poutput, "SCISSOR_RECT", gpu_address);
   poutput.end_block();
}

void
BatchbufferDecoder::
decode_pointer_helper(BatchbufferLoggerOutput &poutput,
                      const char *instruction_name, uint64_t gpu_address)
{
   struct gen_group *g;

   g = gen_spec_find_struct(m_spec, instruction_name);
   if (g) {
      poutput.print_value("Type", instruction_name);
      decode_pointer_helper(poutput, g, gpu_address);
   } else {
      poutput.print_value("Unknown Type", "%s", instruction_name);
   }
}

void
BatchbufferDecoder::
decode_pointer_helper(BatchbufferLoggerOutput &poutput,
                      struct gen_group *g, uint64_t gpu_address)
{
   const uint32_t *p;
   GPUAddressQuery Q;

   p = m_ctx->cpu_mapped<uint32_t>(gpu_address, &Q);
   if (p) {
      int len;
      len = gen_group_get_length(g, p);

      if (len < 0) {
         poutput.print_value("BAD length", "%d", len);
         return;
      }

      if (Q.m_offset_into_gem_bo + len > Q.m_gem_bo->size()) {
         poutput.begin_block("Length to large");
         poutput.print_value("length", "%d", len);
         poutput.print_value("GEM BO offset", "%u", Q.m_offset_into_gem_bo);
         poutput.print_value("GEM BO size", "%u", Q.m_gem_bo->size());
         poutput.end_block();
         return;
      }
   } else {
      poutput.print_value("Bad GPU Address", "0x%012" PRIx64, gpu_address);
      return;
   }

   decode_gen_group(poutput, Q.m_gem_bo, Q.m_offset_into_gem_bo, p, g);
}

void
BatchbufferDecoder::
decode_gen_group(BatchbufferLoggerOutput &poutput,
                 const GEMBufferObject *q, uint64_t dword_offset,
                 const uint32_t *p, struct gen_group *group)
{
   struct gen_field_iterator iter;

   gen_field_iterator_init(&iter, group, p, 0, false);

   do {
      if (!gen_field_is_header(iter.field)) {
         if (iter.struct_desc) {
            int iter_dword = iter.bit / 32;
            uint64_t struct_offset;

            struct_offset = dword_offset + iter_dword;
            poutput.begin_block_value(iter.name, "%s", iter.value);
            decode_gen_group(poutput, q, struct_offset,
                             p + iter_dword, iter.struct_desc);
            poutput.end_block();
         } else {
            poutput.print_value(iter.name, "%s", iter.value);
         }
      }
   } while (gen_field_iterator_next(&iter));
}

void
BatchbufferDecoder::
decode_gpu_command(BatchbufferLoggerOutput &poutput, const GPUCommand &q)
{
   poutput.begin_block(gen_group_get_name(q.inst()));
   decode_gen_group(poutput, q.gem_bo(), q.dword_offset(), q.contents_ptr(), q.inst());
   DetailedDecoder::decode(this, poutput, q);
   poutput.end_block();
}

void
BatchbufferDecoder::
decode_gpu_execute_command(BatchbufferLoggerOutput &poutput, const GPUCommand &q)
{
   poutput.begin_block("Execute GPU command");
   poutput.print_value("Command", "%s", gen_group_get_name(q.inst()));

   decode_gpu_command(poutput, q);
   poutput.begin_block("GPU State");
   m_gpu_state.decode_contents(this, q.gpu_pipeline_type(), poutput);
   poutput.end_block();

   poutput.end_block();
}

void
BatchbufferDecoder::
process_gpu_command(BatchbufferLoggerOutput &poutput,
                    const GPUCommand &q)
{
   enum GPUCommand::gpu_command_type_t tp;

   m_gpu_state.update_state(this, poutput, q);
   tp = q.gpu_command_type();
   switch (tp) {
   case GPUCommand::gpu_command_show_value_with_gpu_state:
      if (poutput) {
         decode_gpu_execute_command(poutput, q);
      }
      break;

   case GPUCommand::gpu_command_show_value_without_gpu_state:
      if (poutput) {
         decode_gen_group(poutput, q.gem_bo(), q.dword_offset(), q.contents_ptr(), q.inst());
         DetailedDecoder::decode(this, poutput, q);
      }
      break;

   default:
      /* nothing */
      break;
   }
}

void
BatchbufferDecoder::
handle_batchbuffer_start(BatchbufferLoggerOutput &dst,
                         const GPUCommand &gpu_command)
{
   uint64_t gpu_address;
   uint32_t batchbuffer_length;

   if (!gpu_command.extract_field_value<uint32_t>("DWord Length", &batchbuffer_length) ||
       batchbuffer_length == 0 ||
       !gpu_command.extract_field_value<uint64_t>("Batch Buffer Start Address", &gpu_address)) {
      return;
   }

   GPUAddressQuery gpu_address_query;
   gpu_address_query = m_ctx->get_memory_at_gpu_address(gpu_address);
   if (!gpu_address_query.m_gem_bo) {
      return;
   }

   uint64_t batchend;
   batchend = gpu_address_query.m_offset_into_gem_bo + 4 * batchbuffer_length;
   if (batchend > gpu_address_query.m_gem_bo->size()) {
      return;
   }

   if (dst) {
      decode_gen_group(dst, gpu_command.gem_bo(), gpu_command.dword_offset(),
                       gpu_command.contents_ptr(), gpu_command.inst());
      dst.begin_block_value("Command Contents", "@%u", gpu_command.dword_offset());
   }

   /* now recurse; */
   BatchbufferLog *sub;
   sub = m_tracker->fetch_or_create(nullptr, gpu_address_query.m_gem_bo->handle());
   sub->emit_log(this, dst, gpu_address_query.m_gem_bo,
                 gpu_address_query.m_offset_into_gem_bo / 4, batchbuffer_length,
                 -1);
   /* BLEH. We are (potentially incorrectly) assuming any batchbuffer is
    * used only once, including sub-batchbuffers.
    */
   m_tracker->remove_batchbuffer_log(sub);

   if (dst) {
      dst.end_block();
   }

}

void
BatchbufferDecoder::
handle_batchbuffer_contents(BatchbufferDecoder *decoder,
                            BatchbufferLoggerOutput &dst,
                            const GEMBufferObject *batchbuffer,
                            uint32_t start, uint32_t end)
{
   if (dst) {
      dst.begin_block_value("GPU commands", "[%u, %u)", start, end);
      dst.print_value("dword start", "%u", start);
      dst.print_value("dword end", "%u", end);
      dst.print_value("dword length", "%u", end - start);
   }

   if (decoder) {
      decoder->absorb_batchbuffer_contents(dst, batchbuffer, start, end);
   }

   if (dst) {
      dst.end_block();
   }
}

void
BatchbufferDecoder::
absorb_batchbuffer_contents(BatchbufferLoggerOutput &dst,
                            const GEMBufferObject *batchbuffer,
                            unsigned int start_dword, unsigned int end_dword)
{
   int length;

   if (start_dword >= end_dword || m_decode_level == no_decode) {
      return;
   }

   for (; start_dword < end_dword;  start_dword += length) {
      GPUCommand q(batchbuffer, start_dword, m_spec);

      m_gpu_command_counter->increment(q);
      length = std::max(1u, q.contents_size());
      if (q.inst()) {
         if (dst) {
            std::ostringstream str;
            dst.begin_block_value(gen_group_get_name(q.inst()), "%u", start_dword);
         }

         if (gen_group_get_opcode(q.inst()) == _MI_BATCH_BUFFER_START) {
            m_gpu_command_counter->add_batch();
            handle_batchbuffer_start(dst, q);
         } else if (m_decode_level >= instruction_details_decode) {
            process_gpu_command(dst, q);
         }

         if (dst) {
            dst.end_block();
         }
      } else if (dst) {
         uint32_t h = q[0];
         uint32_t type = field<uint32_t>(h, 29, 31);
         uint32_t command_opcode, command_subopcode, subtype;

         dst.begin_block_value("Unknown instruction", "%u (0x%08x)",
                               start_dword, q[0]);
         switch(type) {
         case 0:
            command_opcode = field<uint32_t>(h, 23, 28);
            dst.print_value("type", "MI (%d)", type);
            dst.print_value("command_opcode", "0x%02x", command_opcode);
            break;
         case 2:
            command_opcode = field<uint32_t>(h, 22, 28);
            dst.print_value("type", "BLT (%d)", type);
            dst.print_value("command_opcode", "0x%02x", command_opcode);
            break;
         case 3:
            dst.print_value("type", "Render (%d)", type);
            subtype = field<uint32_t>(h, 27, 28);
            command_opcode = field<uint32_t>(h, 24, 26);
            command_subopcode = field<uint32_t>(h, 16, 23);
            dst.print_value("subtype", "0x%01x", subtype);
            dst.print_value("command_opcode", "0x%01x", command_opcode);
            dst.print_value("command_subopcode", "0x%04x", command_subopcode);
            break;
         default:
            dst.print_value("type", "Unknown (%d)", type);
            break;
         }
         dst.print_value("dword length", "%d", q.contents_size());
         dst.end_block();
      }
   }
}

//////////////////////////////////
// ShaderFileList methods
const char*
ShaderFileList::
filename(const void *shader, int pciid, struct gen_disasm *gen_disasm,
         const char *label)
{
   key_type key;
   std::map<key_type, std::string>::iterator iter;
   int shader_sz;

   shader_sz = gen_disasm_assembly_length(gen_disasm, shader, 0);
   _mesa_sha1_compute(shader, shader_sz, key.first.data());
   key.second = label;
   iter = m_files.find(key);
   if (iter != m_files.end()) {
      return iter->second.c_str();
   }

   std::ostringstream str;
   std::string filename;

   str << label << "#" << ++m_count
       << ".pciid.0x" << std::hex << pciid << ".shader_binary";
   filename = str.str();

   std::ofstream shader_file(filename.c_str(),
                             std::ios_base::out | std::ios_base::binary);
   if (!shader_file.is_open()) {
      return nullptr;
   }

   shader_file.write(static_cast<const char*>(shader), shader_sz);
   iter = m_files.insert(std::make_pair(key, filename)).first;

   return iter->second.c_str();
}

const char*
ShaderFileList::
disassembly(const void *shader, int pciid, struct gen_disasm *gen_disasm)
{
   sha1_value key;
   std::map<sha1_value, std::string>::iterator iter;
   int shader_sz;

   shader_sz = gen_disasm_assembly_length(gen_disasm, shader, 0);
   _mesa_sha1_compute(shader, shader_sz, key.data());
   iter = m_disassembly.find(key);
   if (iter != m_disassembly.end()) {
      return iter->second.c_str();
   }

   std::FILE *tmp;
   std::string data;
   long sz, sz_read;

   tmp = std::tmpfile();
   gen_disasm_disassemble(gen_disasm, shader, 0, tmp);
   sz = std::ftell(tmp);
   std::rewind(tmp);
   data.resize(sz + 1, '\0');
   sz_read = std::fread(&data[0], sizeof(char), sz, tmp);
   assert(sz == sz_read);
   (void)sz_read;
   std::fclose(tmp);

   std::string &dst(m_disassembly[key]);
   std::swap(dst, data);
   return dst.c_str();
}

//////////////////////////////////
// BatchbufferLog methods
void
BatchbufferLog::
add_ioctl_log_entry(const std::string &entry)
{
   if (!m_prints.empty()) {
      m_prints.back().add_ioctl_log_entry(entry);
   } else {
      m_orphan_ioctl_log_entries.push_back(entry);
   }
}

void
BatchbufferLog::
add_item(std::shared_ptr<MessageActionBase> p)
{
   if (m_prints.empty()) {
      m_pre_print_items.add_item(p);
   } else {
      m_prints.back().add_item(p);
   }
}

void
BatchbufferLog::
emit_log(BatchbufferLoggerOutput &dst) const
{
   emit_log(nullptr, dst, nullptr, 0, 0, -1);
}

void
BatchbufferLog::
emit_log(BatchbufferDecoder *decoder, BatchbufferLoggerOutput &dst,
         const GEMBufferObject *batchbuffer,
         uint32_t batchbuffer_start, uint32_t batchbuffer_len,
         int bb_id) const
{
   unsigned int last_time(batchbuffer_start);
   unsigned int top_level(dst.current_block_level());

   for(auto iter = m_prints_from_dummy.begin();
       iter != m_prints_from_dummy.end(); ++iter) {
      iter->emit(iter->start_bb_location(), dst, top_level, nullptr, nullptr);
   }

   APIStartCallMarker::print_ioctl_log(m_orphan_ioctl_log_entries, dst);

   if (!m_prints.empty() && m_prints.begin()->start_bb_location() > batchbuffer_start) {
      BatchbufferDecoder::handle_batchbuffer_contents(decoder, dst, batchbuffer,
                                                      batchbuffer_start,
                                                      m_prints.begin()->start_bb_location());
   }

   for(auto iter = m_prints.begin(); iter != m_prints.end(); ++iter) {
      const APIStartCallMarker &entry(*iter);
      auto next_iter(iter);
      uint32_t next_time;

      ++next_iter;
      next_time = (next_iter != m_prints.end()) ?
         next_iter->start_bb_location() :
         batchbuffer_len;
      entry.emit(next_time, dst, top_level, decoder, batchbuffer);
      last_time = next_time;
   }

   if (batchbuffer_len > last_time) {
      if (dst) {
         dst.clear_block_stack(top_level + 1);
      }
      BatchbufferDecoder::handle_batchbuffer_contents(decoder, dst, batchbuffer,
                                                      last_time, batchbuffer_len);
   }

   /* close up all blocks we have left open */
   if (dst) {
      dst.clear_block_stack(top_level);
   }
}

////////////////////////////
// ManagedGEMBuffer methods
ManagedGEMBuffer::
ManagedGEMBuffer(int fd, __u64 sz):
   m_fd(fd), m_handle(0), m_sz(sz),
   m_mapped(nullptr)
{
   struct drm_i915_gem_create create_info;
   int ret;

   create_info.size = sz;
   ret = BatchbufferLogger::local_drm_ioctl(m_fd, DRM_IOCTL_I915_GEM_CREATE, &create_info);
   if (ret == 0) {
      m_handle = create_info.handle;
   }
}

ManagedGEMBuffer::
~ManagedGEMBuffer()
{
   if (m_handle) {
      struct drm_gem_close gem_close;

      unmap();
      gem_close.handle = m_handle;
      gem_close.pad = 0;
      BatchbufferLogger::local_drm_ioctl(m_fd, DRM_IOCTL_GEM_CLOSE, &gem_close);
   }
}

void*
ManagedGEMBuffer::
map_implement(void)
{
   struct drm_i915_gem_mmap map_str;
   int ret;

   if (m_mapped) {
      return m_mapped;
   }

   assert(m_handle != 0);
   std::memset(&map_str, 0, sizeof(map_str));
   map_str.handle = m_handle;
   map_str.offset = 0;
   map_str.size = m_sz;
   ret = BatchbufferLogger::local_drm_ioctl(m_fd, DRM_IOCTL_I915_GEM_MMAP, &map_str);
   if (ret == 0) {
      m_mapped = (void*) map_str.addr_ptr;
   }

   return m_mapped;
}

void
ManagedGEMBuffer::
unmap(void)
{
   if (m_mapped) {
      munmap(m_mapped, m_sz);
      m_mapped = nullptr;
   }
}

////////////////////////////
// PerHWContext methods
PerHWContext::
PerHWContext(uint32_t id):
   i965HWContextData(id)
{}

PerHWContext::
~PerHWContext()
{}

bool
PerHWContext::
update_gem_bo_gpu_address(GEMBufferObject *gem, uint64_t new_address)
{
   std::pair<bool, uint64_t> b;
   b = gem->update_gpu_address(ctx_id(), new_address);
   if (b.first) {
      uint64_t new_end;
      m_gem_bos_by_gpu_address_end.erase(b.second + gem->size());
      new_end = new_address + gem->size();
      m_gem_bos_by_gpu_address_end[new_end] = gem;
   }
   return b.first;
}

GPUAddressQuery
PerHWContext::
get_memory_at_gpu_address(uint64_t address) const
{
   std::map<uint64_t, GEMBufferObject*>::const_iterator iter;
   GPUAddressQuery return_value;

   return_value.m_gem_bo = nullptr;
   return_value.m_offset_into_gem_bo = 0uL;
   /* Get the first BO whose GPU end address is
    * greater than address, thus iter->first > address
    */
   iter = m_gem_bos_by_gpu_address_end.upper_bound(address);
   if (iter == m_gem_bos_by_gpu_address_end.end()) {
      return return_value;
   }

   uint64_t gem_addr;
   gem_addr = iter->second->gpu_address_begin(ctx_id());

   if (gem_addr <= address) {
     return_value.m_gem_bo = iter->second;
     return_value.m_offset_into_gem_bo = address - gem_addr;

     assert(address >= iter->second->gpu_address_begin(ctx_id()));
     assert(address < iter->second->gpu_address_end(ctx_id()));
   }

   return return_value;
}

int
PerHWContext::
pread_buffer(void *dst, uint64_t gpu_address, uint64_t size) const
{
   GPUAddressQuery q;
   q = get_memory_at_gpu_address(gpu_address);

   if (q.m_gem_bo) {
      uint64_t offset;
      offset = q.m_offset_into_gem_bo - gpu_address;
      return q.m_gem_bo->pread_buffer(dst, offset, size);
   } else {
      return -1;
   }
}

template<typename T>
const T*
PerHWContext::
cpu_mapped(uint64_t gpu_address, GPUAddressQuery *q)
{
   GPUAddressQuery Q;

   q = (q) ? q : &Q;
   *q = get_memory_at_gpu_address(gpu_address);
   if (q->m_gem_bo) {
      const void *p;
      p = q->m_gem_bo->cpu_mapped<uint8_t>() + q->m_offset_into_gem_bo;
      return static_cast<const T*>(p);
   } else {
      return nullptr;
   }
}

void
PerHWContext::
drop_gem(GEMBufferObject *p)
{
   if (p->on_gtt(ctx_id())) {
      m_gem_bos_by_gpu_address_end.erase(p->gpu_address_end(ctx_id()));
   }
}

//////////////////////////////
// GEMBufferTracker methods
GEMBufferTracker::
GEMBufferTracker(int fd):
   m_fd(fd),
   m_fd_has_exec_capture(false),
   m_dummy_hw_ctx(0)
{
   struct drm_i915_getparam p;
   int value(0);

   std::memset(&p, 0, sizeof(p));
   p.param = I915_PARAM_HAS_EXEC_CAPTURE;
   p.value = &value;
   if (0 == BatchbufferLogger::local_drm_ioctl(m_fd, DRM_IOCTL_I915_GETPARAM, &p)) {
      m_fd_has_exec_capture = (value != 0);
   }

}

GEMBufferTracker::
~GEMBufferTracker()
{
   for(const auto &value : m_gem_bos_by_handle) {
      delete value.second;
   }
}


void
GEMBufferTracker::
update_gem_bo_gpu_addresses(const struct drm_i915_gem_execbuffer2 *app,
                            unsigned int req, BatchRelocs *out_relocs)
{
   /* TODO: instead of creating an destroying a GEM BO at each call,
    * we should have a pool of GEM BO's to reuse, using the ioctl
    * DRM_I915_GEM_BUSY to decide to reuse a GEM.
    */
   PerHWContext *ctx(fetch_hw_context(app->rsvd1));
   if (!ctx) {
      return;
   }

   ManagedGEMBuffer batch(m_fd, getpagesize());
   if (!batch.handle()) {
      return;
   }
   uint32_t *mapped;
   mapped = batch.map<uint32_t>();
   *mapped = _MI_BATCH_BUFFER_END;
   batch.unmap();

   std::vector<struct drm_i915_gem_exec_object2> objs(app->buffer_count + 1);
   const struct drm_i915_gem_exec_object2 *src_exec_objs;

   src_exec_objs = (struct drm_i915_gem_exec_object2*) (uintptr_t) app->buffers_ptr;
   for (__u32 i = 0; i < app->buffer_count; ++i) {
      /* TODO: we are relying kernel behavior that the kernel
       * places the GEM BO into the PGTT even if relocation list
       * within the drm_i915_gem_exec_object2 is empty; the right
       * thing to do would be to add a reloc for each src_exec_objs
       * into our ManagedGEMBuffer batch
       */
      std::memcpy(&objs[i], &src_exec_objs[i], sizeof(src_exec_objs[i]));
      objs[i].relocation_count = 0;
      objs[i].relocs_ptr = 0;
   }

   if (app->flags & I915_EXEC_BATCH_FIRST) {
      objs.back().handle = objs.front().handle;
      objs.front().handle = batch.handle();
   } else {
      objs.back().handle = batch.handle();
   }

   struct drm_i915_gem_execbuffer2 q;
   int ioctl_error;

   std::memcpy(&q, app, sizeof(q));
   q.buffers_ptr = (__u64) (uintptr_t) &objs[0];
   q.buffer_count = objs.size();
   q.batch_start_offset = 0;
   q.batch_len = batch.size();
   ioctl_error = BatchbufferLogger::local_drm_ioctl(m_fd, req, &q);
   if (ioctl_error != 0) {
      return;
   }

   const struct drm_i915_gem_exec_object2 *lut;
   if (app->flags & I915_EXEC_HANDLE_LUT) {
      lut = src_exec_objs;
   } else {
      lut = nullptr;
   }

   /* Update the GEM BO addresses. */
   for (__u32 i = 0; i < app->buffer_count; ++i) {
      GEMBufferObject *q;

      /* there is no GEM BO address to update for the GEM BO
       * of batch because it lives and dies only here AND
       * it is not from the application/driver.
       */
      if (objs[i].handle == batch.handle()) {
         continue;
      }

      q = fetch_gem_bo(objs[i].handle);
      if (!q) {
         continue;
      }

      ctx->update_gem_bo_gpu_address(q, objs[i].offset);
   }

   /* now with all the GEM BO addresses updated, now add
    * the reloc information
    */
   for (__u32 i = 0; i < app->buffer_count; ++i) {
      GEMBufferObject *q;

      q = fetch_gem_bo(objs[i].handle);
      if (!q) {
         continue;
      }
      out_relocs->add_entries(src_exec_objs[i], q, this, ctx, lut);
   }
}

void
GEMBufferTracker::
add_gem_bo(const struct drm_i915_gem_create &pdata)
{
   GEMBufferObject *p;
   p = new GEMBufferObject(m_fd, pdata);
   m_gem_bos_by_handle[pdata.handle] = p;
}

void
GEMBufferTracker::
add_gem_bo(const struct drm_i915_gem_userptr &pdata)
{
   GEMBufferObject *p;
   p = new GEMBufferObject(m_fd, pdata);
   m_gem_bos_by_handle[pdata.handle] = p;
}

void
GEMBufferTracker::
remove_gem_bo(uint32_t h)
{
   std::map<uint32_t, GEMBufferObject*>::const_iterator iter;
   GEMBufferObject *p;

   iter = m_gem_bos_by_handle.find(h);
   if (iter != m_gem_bos_by_handle.end()) {
      p = iter->second;
      for (auto s : m_hw_contexts) {
         s.second->drop_gem(p);
      }
      delete p;
      m_gem_bos_by_handle.erase(iter);
   }
}

GEMBufferObject*
GEMBufferTracker::
fetch_gem_bo(uint32_t h) const
{
   std::map<uint32_t, GEMBufferObject*>::const_iterator iter;
   iter = m_gem_bos_by_handle.find(h);
   return (iter != m_gem_bos_by_handle.end()) ?
      iter->second :
      nullptr;
}

void
GEMBufferTracker::
add_hw_context(const struct drm_i915_gem_context_create &create)
{
   uint32_t h;
   h = create.ctx_id;
   m_hw_contexts.insert(std::make_pair(h, new PerHWContext(h)));
}

void
GEMBufferTracker::
remove_hw_context(const struct drm_i915_gem_context_destroy &destroy)
{
   auto iter = m_hw_contexts.find(destroy.ctx_id);
   if (iter != m_hw_contexts.end()) {
      delete iter->second;
      m_hw_contexts.erase(iter);
   }
}

PerHWContext*
GEMBufferTracker::
fetch_hw_context(uint32_t h)
{
   if (h == 0) {
      return &m_dummy_hw_ctx;
   }

   auto iter = m_hw_contexts.find(h);
   return (iter != m_hw_contexts.end()) ? iter->second : nullptr;
}

std::pair<bool, GEMBufferObject*>
GEMBufferTracker::
update_gem_bo_gpu_address(uint32_t ctx_id, const struct drm_i915_gem_exec_object2 *p)
{
   return update_gem_bo_gpu_address(fetch_hw_context(ctx_id), p);
}

std::pair<bool, GEMBufferObject*>
GEMBufferTracker::
update_gem_bo_gpu_address(PerHWContext *ctx, const struct drm_i915_gem_exec_object2 *p)
{
   std::map<uint32_t, GEMBufferObject*>::const_iterator iter;
   std::pair<bool, GEMBufferObject*> return_value;

   if (!ctx) {
      return std::make_pair(false, nullptr);
   }

   iter = m_gem_bos_by_handle.find(p->handle);
   if (iter == m_gem_bos_by_handle.end() || !ctx) {
      return std::make_pair(false, nullptr);
   }

   return_value.first = ctx->update_gem_bo_gpu_address(iter->second, p->offset);
   return_value.second = iter->second;
   return return_value;
}

BatchbufferLog*
GEMBufferTracker::
fetch(uint32_t gem_handle)
{
   std::map<uint32_t, BatchbufferLog>::iterator iter;
   iter = m_logs.find(gem_handle);
   return (iter != m_logs.end()) ?
      &iter->second:
      nullptr;
}

BatchbufferLog*
GEMBufferTracker::
fetch_or_create(const void *bb, uint32_t h)
{
   BatchbufferLog *b;
   b = fetch(h);

   if (b == nullptr) {
      std::map<uint32_t, BatchbufferLog>::iterator iter;
      BatchbufferLog m(m_fd, bb, h);

      iter = m_logs.insert(std::make_pair(h, m)).first;
      b = &iter->second;
   }

   return b;
}

void
GEMBufferTracker::
remove_batchbuffer_log(const BatchbufferLog *q)
{
   assert(q != nullptr);
   assert(q == fetch(q->src()->gem_bo));
   m_logs.erase(q->src()->gem_bo);
}

///////////////////////////////
// BatchbufferLogger methods
BatchbufferLogger::
BatchbufferLogger(void):
   m_batchbuffer_state(default_batchbuffer_state_fcn),
   m_active_batchbuffer(default_active_batchbuffer_fcn),
   m_gen_spec(nullptr),
   m_gen_disasm(nullptr),
   m_number_ioctls(0),
   m_dummy(),
   m_output()
{
   static unsigned int creation_count = 0;

   m_creation_ID = creation_count++;

   /* driver interface */
   clear_batchbuffer_log = clear_batchbuffer_log_fcn;
   migrate_batchbuffer = migrate_batchbuffer_fcn;
   add_message = add_message_fcn;
   release_driver = release_driver_fcn;

   /* counter interface */
   create_counter = create_counter_fcn;
   activate_counter = activate_counter_fcn;
   deactivate_counter = deactivate_counter_fcn;
   print_counter = print_counter_fcn;
   reset_counter = reset_counter_fcn;
   release_counter = release_counter_fcn;

   /* application interface */
   pre_call = pre_call_fcn;
   post_call = post_call_fcn;
   begin_session = begin_session_fcn;
   begin_file_session = begin_file_session_fcn;
   end_session = end_session_fcn;
   release_app = release_app_fcn;

   std::string pciid;
   pciid = read_from_environment<std::string>("I965_PCI_ID", std::string());
   if (!pciid.empty()) {
      std::istringstream istr(pciid);
      uint32_t v;
      istr >> std::hex >> v;
      set_pci_id(v);
      printf("Set PCIID to 0x%0x from %s\n", v, pciid.c_str());
   }

   std::string decode_level_str;

   decode_level_str =
      read_from_environment<std::string>("I965_DECODE_LEVEL",
                                         "instruction_details_decode");
   if (decode_level_str == "no_decode") {
      m_decode_level = BatchbufferDecoder::no_decode;
   } else if (decode_level_str == "instruction_decode") {
      m_decode_level = BatchbufferDecoder::instruction_decode;
   } else {
      m_decode_level = BatchbufferDecoder::instruction_details_decode;
   }

   decode_level_str =
      read_from_environment<std::string>("I965_PRINT_RELOC_LEVEL",
                                         "print_reloc_nothing");
   if (decode_level_str == "print_reloc_gem_gpu_updates") {
      m_print_reloc_level = BatchbufferDecoder::print_reloc_gem_gpu_updates;
   } else {
      m_print_reloc_level = BatchbufferDecoder::print_reloc_nothing;
   }

   m_decode_shaders =
      read_from_environment<int>("I965_DECODE_SHADERS", 1);

   m_process_execbuffers_before_ioctl =
      read_from_environment<int>("I965_DECODE_BEFORE_IOCTL", 0);

   m_emit_capture_execobj_batchbuffer_identifier =
      read_from_environment<int>("I965_EMIT_CAPTURE_EXECOBJ_BATCHBUFFER_IDENTIFIER", 0);

   m_organize_by_ioctls =
      read_from_environment<int>("I965_ORGANIZE_BY_IOCTL", 1);
}

BatchbufferLogger::
~BatchbufferLogger()
{
   m_output.close_all_sessions();

   for (const auto &v : m_gem_buffer_trackers) {
      delete v.second;
   }

   emit_total_stats();
   if (m_gen_disasm) {
      gen_disasm_destroy(m_gen_disasm);
   }
}

void
BatchbufferLogger::
emit_total_stats(void)
{
   const char *emit_stats;

   emit_stats = std::getenv("I965_EMIT_TOTAL_STATS");
   if (emit_stats) {
      /* Kind of a hack; what we do is that we init tmp
       * to refer to a GPUCommandCounter that is set as zero,
       * and after opening the file, we assign zero the
       * values in m_gpu_command_counter which then triggers
       * tmp to print the values of m_gpu_command_counter
       */
      GPUCommandCounter zero;
      BatchbufferLoggerSession *s;
      BatchbufferLoggerOutput tmp;
      std::FILE *file;
      struct i965_batchbuffer_logger_session_params params;

      if (m_creation_ID == 0) {
         file = std::fopen(emit_stats, "w");
      } else {
         std::ostringstream str;
         str << emit_stats << "-" << m_creation_ID;
         file = std::fopen(str.str().c_str(), "w");
      }

      if (!file) {
         return;
      }

      params.client_data = file;
      params.write = &BatchbufferLoggerSession::write_file;
      params.close = &BatchbufferLoggerSession::close_file;
      params.pre_execbuffer2_ioctl = &BatchbufferLoggerSession::flush_file;
      params.post_execbuffer2_ioctl = &BatchbufferLoggerSession::flush_file;
      s = new BatchbufferLoggerSession(params, &zero, false);

      zero = m_gpu_command_counter;
      tmp.remove_session(s);
      delete s;
   }
}

void
BatchbufferLogger::
clear_batchbuffer_log_fcn(struct i965_batchbuffer_logger *pthis,
                          int fd, uint32_t gem_bo)
{
   BatchbufferLogger *R;
   R = static_cast<BatchbufferLogger*>(pthis);

   R->m_mutex.lock();

   BatchbufferLog *bb;
   bb = R->fetch_batchbuffer_log(fd, gem_bo);
   if (bb) {
      R->gem_buffer_tracker(bb->src()->fd)->remove_batchbuffer_log(bb);
   }

   R->m_mutex.unlock();
}

void
BatchbufferLogger::
migrate_batchbuffer_fcn(struct i965_batchbuffer_logger *pthis,
                        const struct i965_logged_batchbuffer *from,
                        const struct i965_logged_batchbuffer *to)
{
   BatchbufferLogger *R;
   R = static_cast<BatchbufferLogger*>(pthis);

   R->m_mutex.lock();

   BatchbufferLog *log_from;
   log_from = R->fetch_batchbuffer_log(from->fd, from->gem_bo);
   if (log_from) {
      BatchbufferLog *log_to;
      log_to = R->fetch_or_create_batchbuffer_log(to);
      log_to->absorb_log(log_from);
      R->remove_batchbuffer_log(log_from);
   }

   R->m_mutex.unlock();
}

void
BatchbufferLogger::
add_message_fcn(struct i965_batchbuffer_logger *pthis,
                const struct i965_logged_batchbuffer *dst,
                const char *fmt)
{
   BatchbufferLogger *R;
   R = static_cast<BatchbufferLogger*>(pthis);

   R->m_mutex.lock();

   BatchbufferLog *log_dst(nullptr);
   uint32_t time_of_print(0);

   if (dst) {
      log_dst = R->fetch_or_create_batchbuffer_log(dst);
   }

   if (!log_dst) {
      log_dst = &R->m_dummy;
   }

   assert(log_dst);
   if (log_dst != &R->m_dummy) {
      time_of_print = R->m_batchbuffer_state(log_dst->src());
   }

   std::shared_ptr<MessageActionBase> p;
   p = std::make_shared<MessageItem>(time_of_print, fmt);
   log_dst->add_item(p);

   R->m_mutex.unlock();
}

void
BatchbufferLogger::
release_driver_fcn(struct i965_batchbuffer_logger *pthis)
{
   release();
}

struct i965_batchbuffer_counter
BatchbufferLogger::
create_counter_fcn(struct i965_batchbuffer_logger *pthis,
                   const char *filename)
{
   struct i965_batchbuffer_counter return_value;
   return_value.opaque = new GPUCommandCounter();
   return return_value;
}

void
BatchbufferLogger::
activate_counter_fcn(struct i965_batchbuffer_logger *pthis,
                     struct i965_batchbuffer_counter counter)
{
   BatchbufferLogger *R;
   GPUCommandCounter *Q;
   BatchbufferLog *log_dst(nullptr);
   uint32_t time_of_print(0);

   R = static_cast<BatchbufferLogger*>(pthis);
   Q = static_cast<GPUCommandCounter*>(counter.opaque);
   R->m_mutex.lock();

   log_dst = R->fetch_or_create_batchbuffer_log();
   assert(log_dst);

   if (log_dst != &R->m_dummy) {
      time_of_print = R->m_batchbuffer_state(log_dst->src());
   }

   std::shared_ptr<MessageActionBase> p;
   p = std::make_shared<ActivateCounter>(time_of_print, &R->m_active_counters, Q);
   log_dst->add_item(p);

   R->m_mutex.unlock();
}

void
BatchbufferLogger::
deactivate_counter_fcn(struct i965_batchbuffer_logger *pthis,
                       struct i965_batchbuffer_counter counter)
{
   BatchbufferLogger *R;
   GPUCommandCounter *Q;
   BatchbufferLog *log_dst(nullptr);
   uint32_t time_of_print(0);

   R = static_cast<BatchbufferLogger*>(pthis);
   Q = static_cast<GPUCommandCounter*>(counter.opaque);
   R->m_mutex.lock();

   log_dst = R->fetch_or_create_batchbuffer_log();
   assert(log_dst);

   if (log_dst != &R->m_dummy) {
      time_of_print = R->m_batchbuffer_state(log_dst->src());
   }

   std::shared_ptr<MessageActionBase> p;
   p = std::make_shared<DeactivateCounter>(time_of_print, &R->m_active_counters, Q);
   log_dst->add_item(p);

   R->m_mutex.unlock();
}

void
BatchbufferLogger::
print_counter_fcn(struct i965_batchbuffer_logger *pthis,
                  struct i965_batchbuffer_counter counter,
                  const char *txt)
{
   BatchbufferLogger *R;
   GPUCommandCounter *Q;
   BatchbufferLog *log_dst(nullptr);
   uint32_t time_of_print(0);

   R = static_cast<BatchbufferLogger*>(pthis);
   Q = static_cast<GPUCommandCounter*>(counter.opaque);
   R->m_mutex.lock();

   log_dst = R->fetch_or_create_batchbuffer_log();
   assert(log_dst);

   if (log_dst != &R->m_dummy) {
      time_of_print = R->m_batchbuffer_state(log_dst->src());
   }

   std::shared_ptr<MessageActionBase> p;
   p = std::make_shared<PrintCounter>(time_of_print, txt, Q);
   log_dst->add_item(p);

   R->m_mutex.unlock();
}

void
BatchbufferLogger::
reset_counter_fcn(struct i965_batchbuffer_logger *pthis,
                  struct i965_batchbuffer_counter counter)
{
   BatchbufferLogger *R;
   GPUCommandCounter *Q;
   BatchbufferLog *log_dst(nullptr);
   uint32_t time_of_print(0);

   R = static_cast<BatchbufferLogger*>(pthis);
   Q = static_cast<GPUCommandCounter*>(counter.opaque);
   R->m_mutex.lock();

   log_dst = R->fetch_or_create_batchbuffer_log();
   assert(log_dst);
   if (log_dst != &R->m_dummy) {
      time_of_print = R->m_batchbuffer_state(log_dst->src());
   }

   std::shared_ptr<MessageActionBase> p;
   p = std::make_shared<ResetCounter>(time_of_print, Q);
   log_dst->add_item(p);

   R->m_mutex.unlock();
}

void
BatchbufferLogger::
release_counter_fcn(struct i965_batchbuffer_logger *pthis,
                    struct i965_batchbuffer_counter counter)
{
   BatchbufferLogger *R;
   GPUCommandCounter *Q;
   BatchbufferLog *log_dst(nullptr);
   uint32_t time_of_print(0);

   R = static_cast<BatchbufferLogger*>(pthis);
   Q = static_cast<GPUCommandCounter*>(counter.opaque);
   R->m_mutex.lock();

   log_dst = R->fetch_or_create_batchbuffer_log();
   assert(log_dst);
   if (log_dst != &R->m_dummy) {
      time_of_print = R->m_batchbuffer_state(log_dst->src());
   }

   std::shared_ptr<MessageActionBase> p;
   p = std::make_shared<ReleaseCounter>(time_of_print, &R->m_active_counters, Q);
   log_dst->add_item(p);

   R->m_mutex.unlock();
}

void
BatchbufferLogger::
pre_call_fcn(struct i965_batchbuffer_logger_app *pthis,
             unsigned int call_id,
             const char *call_detailed,
             const char *fcn_name)
{
   BatchbufferLogger *R;
   BatchbufferLog *bb;
   uint32_t time_of_print(0);

   R = static_cast<BatchbufferLogger*>(pthis);

   R->m_mutex.lock();
   bb = R->fetch_or_create_batchbuffer_log();
   if (bb != &R->m_dummy) {
      time_of_print = R->m_batchbuffer_state(bb->src());
   }
   bb->add_call_marker(R->m_dummy, call_id, fcn_name,
                       call_detailed, time_of_print);
   R->m_mutex.unlock();
}

void
BatchbufferLogger::
post_call_fcn(struct i965_batchbuffer_logger_app *pthis,
              unsigned int call_id)
{
}

struct i965_batchbuffer_logger_session
BatchbufferLogger::
begin_file_session_fcn(struct i965_batchbuffer_logger_app *pthis,
                       const char *filename)
{
   std::FILE *file;

   file = std::fopen(filename, "w");
   if (!file) {
      struct i965_batchbuffer_logger_session p;
      p.opaque = nullptr;
      return p;
   }

   struct i965_batchbuffer_logger_session_params params;
   params.client_data = file;
   params.write = &BatchbufferLoggerSession::write_file;
   params.close = &BatchbufferLoggerSession::close_file;
   params.pre_execbuffer2_ioctl = &BatchbufferLoggerSession::flush_file;
   params.post_execbuffer2_ioctl = &BatchbufferLoggerSession::flush_file;
   return begin_session_fcn(pthis, &params);
}

struct i965_batchbuffer_logger_session
BatchbufferLogger::
begin_session_fcn(struct i965_batchbuffer_logger_app *pthis,
                  const struct i965_batchbuffer_logger_session_params *params)
{
   BatchbufferLogger *R;
   struct i965_batchbuffer_logger_session p;
   BatchbufferLoggerSession *S;

   R = static_cast<BatchbufferLogger*>(pthis);
   R->m_mutex.lock();
   S = new BatchbufferLoggerSession(*params, &R->m_gpu_command_counter, true);
   R->m_output.add_session(S);
   p.opaque = S;
   R->m_mutex.unlock();

   return p;
}

void
BatchbufferLogger::
end_session_fcn(struct i965_batchbuffer_logger_app *pthis,
                struct i965_batchbuffer_logger_session p)
{
   BatchbufferLogger *R;
   BatchbufferLoggerSession *s;

   R = static_cast<BatchbufferLogger*>(pthis);
   s = static_cast<BatchbufferLoggerSession*>(p.opaque);

   R->m_mutex.lock();

   R->m_output.remove_session(s);
   delete s;

   R->m_mutex.unlock();
}

void
BatchbufferLogger::
release_app_fcn(struct i965_batchbuffer_logger_app *pthis)
{
   release();
}

GEMBufferTracker*
BatchbufferLogger::
gem_buffer_tracker(int fd)
{
   GEMBufferTracker *q(nullptr);
   std::map<int, GEMBufferTracker*>::iterator iter;

   iter = m_gem_buffer_trackers.find(fd);
   if (iter != m_gem_buffer_trackers.end()) {
      q = iter->second;
   } else if (fd != -1) {
      q = new GEMBufferTracker(fd);
      m_gem_buffer_trackers[fd] = q;
   }

   return q;
}

BatchbufferLog*
BatchbufferLogger::
fetch_batchbuffer_log(int fd, uint32_t gem_bo)
{
   /* We do NOT want to create a BatchbufferLog
    * object, thus we use the call that only fetches
    * and does not create.
    */
   GEMBufferTracker *tracker;

   tracker = gem_buffer_tracker(fd);
   return (tracker) ?
      tracker->fetch(gem_bo) :
      nullptr;
}


BatchbufferLog*
BatchbufferLogger::
fetch_or_create_batchbuffer_log(const struct i965_logged_batchbuffer *bb)
{
   int fd;
   GEMBufferTracker *tracker;

   fd = (bb != nullptr && bb->gem_bo != 0) ?
      bb->fd :
      -1;

   tracker = gem_buffer_tracker(fd);
   return (tracker) ?
      tracker->fetch_or_create(bb->driver_data, bb->gem_bo) :
      &m_dummy;
}

BatchbufferLog*
BatchbufferLogger::
fetch_or_create_batchbuffer_log(void)
{
   struct i965_logged_batchbuffer bb;
   m_active_batchbuffer(&bb);
   return fetch_or_create_batchbuffer_log(&bb);
}

void
BatchbufferLogger::
remove_batchbuffer_log(const BatchbufferLog *q)
{
   GEMBufferTracker *tracker;
   tracker = gem_buffer_tracker(q->src()->fd);
   if (tracker) {
      tracker->remove_batchbuffer_log(q);
   }
}

void
BatchbufferLogger::
print_ioctl_message(const std::string &msg)
{
   std::ostringstream str;
   str << "(non-execbuffer2)IOCTL #" << ++m_number_ioctls;
   m_output.print_value(str.str().c_str(), "%s", msg.c_str());
}

ManagedGEMBuffer*
BatchbufferLogger::
pre_process_ioctl(int fd, unsigned long request, void *argp)
{
   m_mutex.lock();

   ManagedGEMBuffer *return_value(nullptr);
   GEMBufferTracker *tracker;

   tracker = gem_buffer_tracker(fd);
   if (!tracker) {
      return return_value;
   }

   switch (request) {
   case DRM_IOCTL_I915_GEM_EXECBUFFER2:
   case DRM_IOCTL_I915_GEM_EXECBUFFER2_WR: {
      if (m_process_execbuffers_before_ioctl) {
         struct drm_i915_gem_execbuffer2 *execbuffer2 =
            (struct drm_i915_gem_execbuffer2*) argp;

         if (tracker->fd_has_exec_capture() &&
             m_emit_capture_execobj_batchbuffer_identifier &&
             (execbuffer2->flags & I915_EXEC_BATCH_FIRST) != 0) {
            /* create a ManagedGEMBuffer,  whose contents is (as a string)
             * the execbuffer2 ioctl ID (these values are visible in the log)
             */
            char buffer[4096];
            unsigned int len;

            len = std::snprintf(buffer, sizeof(buffer), "(execbuffer2) IOCTL #%d", m_number_ioctls);
            assert(len < sizeof(buffer));

            return_value = new ManagedGEMBuffer(fd, len);
            std::memcpy(return_value->map<char>(), buffer, len);
            return_value->unmap();
         }
         BatchRelocs relocs(m_gen_spec);

         tracker->update_gem_bo_gpu_addresses(execbuffer2, request, &relocs);
         BatchbufferDecoder decoder(m_decode_level, m_print_reloc_level,
                                    m_decode_shaders, m_organize_by_ioctls,
                                    m_gen_spec, m_gen_disasm,
                                    m_pci_id, tracker,
                                    false, relocs,
                                    &m_gpu_command_counter,
                                    &m_shader_filelist, execbuffer2);

         assert(decoder.batchbuffer_log());
         m_output.pre_execbuffer2_ioctl(m_number_ioctls);
         decoder.emit_log(m_output, m_number_ioctls++);
         m_output.post_execbuffer2_ioctl(m_number_ioctls);
      }
   } break;

   }

   return return_value;
}

void
BatchbufferLogger::
post_process_ioctl(int ioctl_return_code, int fd, unsigned long request,
                   void *argp)
{
   if (ioctl_return_code != 0) {
      m_mutex.unlock();
      return;
   }

   GEMBufferTracker *tracker;
   BatchbufferLog *bb;
   struct i965_logged_batchbuffer driver_bb;

   tracker = gem_buffer_tracker(fd);
   m_active_batchbuffer(&driver_bb);
   if (driver_bb.fd == fd) {
      bb = tracker->fetch_or_create(driver_bb.driver_data,
                                    driver_bb.gem_bo);
   } else {
      bb = &m_dummy;
   }

   switch(request) {
   case DRM_IOCTL_I915_GEM_CREATE: {
      struct drm_i915_gem_create *create;

      create = (struct drm_i915_gem_create*) argp;
      tracker->add_gem_bo(*create);

      std::ostringstream ostr;
      ostr << "Create GEM BO fd = " << std::dec << fd
           << ", size = " << create->size
           << ", handle = " << create->handle;

      if (m_organize_by_ioctls) {
         print_ioctl_message(ostr.str());
      } else {
         bb->add_ioctl_log_entry(ostr.str());
      }
   } break;

   case DRM_IOCTL_I915_GEM_USERPTR: {
      struct drm_i915_gem_userptr *create;

      create = (struct drm_i915_gem_userptr*) argp;
      tracker->add_gem_bo(*create);

      std::ostringstream ostr;
      ostr << "Create GEM BO-userptr fd = " << std::dec << fd
           << ", user_size = " << create->user_size
           << ", user_ptr = " << create->user_ptr
           << ", handle = " << create->handle;

      if (m_organize_by_ioctls) {
         print_ioctl_message(ostr.str());
      } else {
         bb->add_ioctl_log_entry(ostr.str());
      }
   } break;

   case DRM_IOCTL_GEM_CLOSE: {
      struct drm_gem_close *cmd;
      std::ostringstream ostr;

      cmd = (struct drm_gem_close *) argp;
      tracker->remove_gem_bo(cmd->handle);

      ostr << "Remove GEM BO fd = " << fd
           << ", handle = " << cmd->handle;

      if (m_organize_by_ioctls) {
         print_ioctl_message(ostr.str());
      } else {
         bb->add_ioctl_log_entry(ostr.str());
      }
   } break;

   case DRM_IOCTL_I915_GEM_CONTEXT_CREATE: {
      struct drm_i915_gem_context_create *create_hw_ctx;

      create_hw_ctx = (struct drm_i915_gem_context_create*)argp;
      tracker->add_hw_context(*create_hw_ctx);

      std::ostringstream ostr;
      ostr << "Create GEM HW context, fd = " << std::dec << fd
           << ", handle = " << create_hw_ctx->ctx_id;

      if (m_organize_by_ioctls) {
         print_ioctl_message(ostr.str());
      } else {
         bb->add_ioctl_log_entry(ostr.str());
      }
   } break;

   case DRM_IOCTL_I915_GEM_CONTEXT_DESTROY: {
      struct drm_i915_gem_context_destroy *destroy_hw_ctx;

      destroy_hw_ctx = (struct drm_i915_gem_context_destroy*)argp;
      tracker->remove_hw_context(*destroy_hw_ctx);

      std::ostringstream ostr;
      ostr << "Destroy GEM HW context, fd = " << std::dec << fd
           << ", handle = " << destroy_hw_ctx->ctx_id;

      if (m_organize_by_ioctls) {
         print_ioctl_message(ostr.str());
      } else {
         bb->add_ioctl_log_entry(ostr.str());
      }
   } break;

   case DRM_IOCTL_I915_GEM_EXECBUFFER: {
      std::fprintf(stderr, "Warning: old school DRM_IOCTL_I915_GEM_EXECBUFFER\n");
   } break;

   case DRM_IOCTL_I915_GEM_EXECBUFFER2:
   case DRM_IOCTL_I915_GEM_EXECBUFFER2_WR: {
      if (!m_process_execbuffers_before_ioctl) {
         struct drm_i915_gem_execbuffer2 *execbuffer2 =
            (struct drm_i915_gem_execbuffer2*) argp;
         BatchRelocs relocs(m_gen_spec);
         BatchbufferDecoder decoder(m_decode_level, m_print_reloc_level,
                                    m_decode_shaders, m_organize_by_ioctls,
                                    m_gen_spec, m_gen_disasm,
                                    m_pci_id, tracker,
                                    true, relocs,
                                    &m_gpu_command_counter,
                                    &m_shader_filelist, execbuffer2);

         assert(decoder.batchbuffer_log());
         m_output.pre_execbuffer2_ioctl(m_number_ioctls);
         decoder.emit_log(m_output, m_number_ioctls++);
         m_output.post_execbuffer2_ioctl(m_number_ioctls);
      }
   } break;

   } //of switch(request)

   m_mutex.unlock();
}

int
BatchbufferLogger::
local_drm_ioctl(int fd, unsigned long request, void *argp)
{
   int ret;

   do {
      ret = ioctl(fd, request, argp);
   } while (ret == -1 && (errno == EINTR || errno == EAGAIN));

   return ret;
}

static pthread_mutex_t i965_batchbuffer_logger_acquire_mutex =
   PTHREAD_MUTEX_INITIALIZER;
static int i965_batchbuffer_logger_acquire_ref_count = 0;
static BatchbufferLogger *i965_batchbuffer_logger_object = nullptr;

BatchbufferLogger*
BatchbufferLogger::
acquire(void)
{
   pthread_mutex_lock(&i965_batchbuffer_logger_acquire_mutex);

   if (!i965_batchbuffer_logger_object) {
      i965_batchbuffer_logger_object = new BatchbufferLogger();
   }
   ++i965_batchbuffer_logger_acquire_ref_count;

   pthread_mutex_unlock(&i965_batchbuffer_logger_acquire_mutex);

   return i965_batchbuffer_logger_object;
}

void
BatchbufferLogger::
release(void)
{
   pthread_mutex_lock(&i965_batchbuffer_logger_acquire_mutex);

   --i965_batchbuffer_logger_acquire_ref_count;
   i965_batchbuffer_logger_object->emit_total_stats();
   if (i965_batchbuffer_logger_acquire_ref_count == 0) {
      delete i965_batchbuffer_logger_object;
      i965_batchbuffer_logger_object = nullptr;
   }

   pthread_mutex_unlock(&i965_batchbuffer_logger_acquire_mutex);
}

void
BatchbufferLogger::
set_pci_id(int pci_id)
{
   int old_pci_id;

   m_mutex.lock();
   old_pci_id = m_pci_id;
   if (gen_get_device_info(pci_id, &m_dev_info)) {
      m_pci_id = pci_id;
      m_gen_spec = gen_spec_load(&m_dev_info);

      if (m_gen_disasm && old_pci_id != m_pci_id) {
         gen_disasm_destroy(m_gen_disasm);
         m_gen_disasm = nullptr;
      }

      if (m_gen_disasm == nullptr) {
         m_gen_disasm = gen_disasm_create(&m_dev_info);
      }
   }

   m_mutex.unlock();
}

void
BatchbufferLogger::
set_driver_funcs(i965_logged_batchbuffer_state f1,
                 i965_active_batchbuffer f2)
{
   m_mutex.lock();
   m_batchbuffer_state = f1;
   m_active_batchbuffer = f2;
   m_mutex.unlock();
}

/* Replacing ioctl with like that found in aubdump of IGT
 * does not work with apitrace; some of the ioctls are
 * picked up, but not all. This appears to only happen on
 * apitrace (and its glretrace program). I have no idea why
 * replacing ioctl does not work, but replacing drmIoctl does
 * work.
 */
extern "C"
int
drmIoctl(int fd, unsigned long request, void *arg)
{
   static ForceExistanceOfBatchbufferLogger bf;
   std::vector<struct drm_i915_gem_exec_object2> exec_objs;
   struct drm_i915_gem_execbuffer2 with_capture;
   int return_value;
   ManagedGEMBuffer *gem_capture(nullptr);

   pthread_mutex_lock(&i965_batchbuffer_logger_acquire_mutex);

   if (i965_batchbuffer_logger_object) {
      gem_capture = i965_batchbuffer_logger_object->pre_process_ioctl(fd, request, arg);
   }

   if (gem_capture) {
      struct drm_i915_gem_exec_object2 *src_objs;

      std::memcpy(&with_capture, arg, sizeof(with_capture));

      src_objs = (struct drm_i915_gem_exec_object2*)(uintptr_t) with_capture.buffers_ptr;
      exec_objs.resize(with_capture.buffer_count + 1);
      std::copy(src_objs, src_objs + with_capture.buffer_count, exec_objs.begin());

      exec_objs.back().handle = gem_capture->handle();
      exec_objs.back().relocation_count = 0;
      exec_objs.back().relocs_ptr = 0;
      exec_objs.back().alignment = 64;
      exec_objs.back().offset = 0;
      exec_objs.back().flags = EXEC_OBJECT_CAPTURE;
      exec_objs.back().rsvd1 = 0;
      exec_objs.back().rsvd2 = 0;

      ++with_capture.buffer_count;
      with_capture.buffers_ptr = (uintptr_t) &exec_objs[0];
      arg = &with_capture;
   }
   return_value = BatchbufferLogger::local_drm_ioctl(fd, request, arg);

   if (i965_batchbuffer_logger_object) {
      i965_batchbuffer_logger_object->post_process_ioctl(return_value, fd,
                                                         request, arg);
   }

   if (gem_capture) {
      delete gem_capture;
   }

   pthread_mutex_unlock(&i965_batchbuffer_logger_acquire_mutex);

   return return_value;
}

//////////////////////////////////////////
// exported symbols for application integration
extern "C"
struct i965_batchbuffer_logger_app*
i965_batchbuffer_logger_app_acquire(void)
{
   BatchbufferLogger *R;
   R = BatchbufferLogger::acquire();
   return R;
}

///////////////////////////////////////////
// exported symbols for 3D driver integration
extern "C"
struct i965_batchbuffer_logger*
i965_batchbuffer_logger_acquire(int pci_id,
                                i965_logged_batchbuffer_state f1,
                                i965_active_batchbuffer f2)
{
   BatchbufferLogger *R;
   R = BatchbufferLogger::acquire();
   R->set_driver_funcs(f1, f2);
   R->set_pci_id(pci_id);
   return R;
}

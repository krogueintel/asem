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

#ifndef I965_BATCHBUFFER_LOGGER_APP_H
#define I965_BATCHBUFFER_LOGGER_APP_H

#include <stdint.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Enumeration value specifying the kind of data to write;
 * Data written to a a i965_batchbuffer_logger_session
 * can mark the start of a block, then end of a block
 * of item within a block. Eac type has an optional
 * name and value strings.
 */
enum i965_batchbuffer_logger_message_type_t {
   I965_BATCHBUFFER_LOGGER_MESSAGE_BLOCK_BEGIN, /* start of a block */
   I965_BATCHBUFFER_LOGGER_MESSAGE_BLOCK_END,   /* end of a block */
   I965_BATCHBUFFER_LOGGER_MESSAGE_VALUE,       /* value in block */
};

/**
 * An i965_batchbuffer_logger_session_params represents how
 * to emit batchbuffer logger data. The batchbuffer logger
 * is controlled by the following environmental variables:
 * - I965_DECODE_BEFORE_IOCTL if non-zero, emit batchbuffer log data
 *                            BEFORE calling the kernel ioctl.
 * - I965_EMIT_TOTAL_STATS gives a filename to which to emit the total
 *                         counts and lengths of GPU commands emitted
 * - I965_PCI_ID pci_id Give a hexadecimal value of the PCI ID value for
 *                      the GPU the BatchbufferLogger to decode for; this
 *                      value is used if and only if the driver fails to
 *                      tell the BatchbufferLogger a valid PCI ID value to
 *                      use
 * - I965_DECODE_LEVEL controls the level of batchbuffer decoding
 *    - no_decode do not decode batchbuffer at all
 *    - instruction_decode decode instruction name only
 *    - instruction_details_decode decode instruction contents
 * - I965_PRINT_RELOC_LEVEL controls at what level to print reloc data
 *    - print_reloc_nothing do not print reloc data
 *    - print_reloc_gem_gpu_updates print reloc data GEM by GEM
 * - I965_DECODE_SHADERS if set and is 0, shader binaries are written to
 *                       file;  otherwise their disassembly is emitted
 *                       in each session
 * - I965_DECODE_BEFORE_IOCTL if set and set to non-zero, the logger will
 *                            decode the contents of a batchbuffer BEFORE
 *                            the batchbuffer is sent to the kernel
 * - I965_EMIT_CAPTURE_EXECOBJ_BATCHBUFFER_IDENTIFIER if set and set to
 *                                                    non-zero, if the kernel
 *                                                    supports EXEC_CAPTURE,
 *                                                    for those execbuffer2
 *                                                    commands where the batch
 *                                                    buffer is the first bo,
 *                                                    the logger will append a
 *                                                    an additional exec_object2
 *                                                    which will hold a string
 *                                                    to identifying the ioctl
 *                                                    ID of the batchbuffer
 * - I965_ORGANIZE_BY_IOCTL if not set or is set to non-zero, the logger will
 *                          organize the logs emitted by ioctl instead of by
 *                          API calls.
 */
struct i965_batchbuffer_logger_session_params {
   /**
    * Client data opaque pointer passed back to
    * function callbacks.
    */
   void *client_data;

   /**
    * Function called by i965_batchbuffer_logger_app to write
    * data for the sessions.
    * \param client_data the pointer value in
    *                    i965_batchbuffer_logger_session_params::client_data
    * \param tp the message type
    * \param name of the data
    * \param name_length length of the name data
    * \param value of the data
    * \param value_length length of the value data
    */
   void (*write)(void *client_data,
                 enum i965_batchbuffer_logger_message_type_t tp,
                 const void *name, uint32_t name_length,
                 const void *value, uint32_t value_length);

   /**
    * This function is called just before the decoding of a batchbuffer.
    * \param client_data the pointer value in
    *                    i965_batchbuffer_logger_session_params::client_data
    * \param id The unique ID of the execbuffer2 ioctl, this ID
    *           is written in the block information of each execbuffer2
    *           ioctl via write().
    */
   void (*pre_execbuffer2_ioctl)(void *client_data, unsigned int id);

   /**
    * This function is called just after the decoding of a batchbuffer.
    * \param client_data the pointer value in
    *                    i965_batchbuffer_logger_session_params::client_data
    * \param id The unique ID of the execbuffer2 ioctl, this ID
    *           is written in the block information of each execbuffer2
    *           ioctl via write().
    */
   void (*post_execbuffer2_ioctl)(void *client_data, unsigned int id);

   /**
    * Function called by i965_batchbuffer_logger_app to close,
    * i.e. delete the session.
    */
   void (*close)(void *client_data);
};

/**
 * An i965_batchbuffer_logger_session represents a logging
 * session created; the batchbuffer logger can have multiple
 * sessions active simutaneously.
 */
struct i965_batchbuffer_logger_session {
   void *opaque;
};

/**
 * An i965_batchbuffer_logger_app represents the hooking
 * of an application into an i965_batchbuffer_logger
 */
struct i965_batchbuffer_logger_app {
  /**
   * To be called by the app before a GL/GLES or GLX/EGL call
   */
  void (*pre_call)(struct i965_batchbuffer_logger_app*,
                   unsigned int call_id, const char *call_detailed,
                   const char *fcn_name);

  /**
   * To be called by the app after a GL/GLES or GLX/EGL call
   */
  void (*post_call)(struct i965_batchbuffer_logger_app*,
                    unsigned int call_id);

  /**
   * To be called by the app to start a session; multiple sessions
   * can be active simutaneously.
   */
  struct i965_batchbuffer_logger_session (*begin_session)(struct i965_batchbuffer_logger_app*,
                                                          const struct i965_batchbuffer_logger_session_params*);

  /**
   * Provided as a conveniance; to be called by an app to log
   * contents to a named file.
   */
   struct i965_batchbuffer_logger_session (*begin_file_session)(struct i965_batchbuffer_logger_app*,
                                                                const char *filename);

  /**
   * To be called by the app to end a session.
   */
  void (*end_session)(struct i965_batchbuffer_logger_app*,
                      struct i965_batchbuffer_logger_session);

  /**
   * To be called by the app to release the i965_batchbuffer_logger_app
   */
  void (*release_app)(struct i965_batchbuffer_logger_app*);
};

/**
 * Call to acquire a hande to (the) i965_batchbuffer_logger_app
 */
struct i965_batchbuffer_logger_app*
i965_batchbuffer_logger_app_acquire(void);



#ifdef __cplusplus
}
#endif

#endif

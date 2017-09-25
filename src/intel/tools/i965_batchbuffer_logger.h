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

#ifndef I965_BATCHBUFFER_LOGGER_DRIVER_H
#define I965_BATCHBUFFER_LOGGER_DRIVER_H

#include <stdint.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * i965_batchbuffer_logger tracks a batchbuffers
 * by the pair (GEM-BO handle, File descriptor)
 * pair.
 */
struct i965_logged_batchbuffer {
   /**
    * GEM BO of the batch buffer, this is the BO
    * sent to kernel to execute commands on the
    * GPU
    */
   uint32_t gem_bo;

   /**
    * The file descriptor of the GEM BO
    */
   int fd;

   /**
    * Opaque pointer used by the driver associated
    * to the batch buffer; an i965_batchbuffer_logger
    * does NOT use this value to identify a batchbuffer.
    * It is for the driver to use to help it compute
    * where it is in a specified batchbuffer.
    */
   const void *driver_data;
};

/**
 * A counter is to be used by a driver to provide counts of elements
 * added to a batchbuffer when the counter is active. Actions acting
 * on a counter are placed on the active batchbuffer of the calling
 * thread, not executed immediately. The actual counting (in addition
 * to activation, deactivation, reset, and relase) are performed when
 * the batchbuffer is intercepted by the logger.
 */
struct i965_batchbuffer_counter {
   void *opaque;
};

/**
 * An i965_batchbuffer_logger object represents the hooking
 * of a GPU driver.
 */
struct i965_batchbuffer_logger {
   /**
    * To be called by the driver to instruct the batchbuffer logger
    * to clear the log associated to a GEM BO from an FD.
    */
   void (*clear_batchbuffer_log)(struct i965_batchbuffer_logger *logger,
                                 int fd, uint32_t gem_bo);

   /**
    * To be called by the driver if it migrates commands from one
    * batchbuffer to another batchbuffer.
    */
   void (*migrate_batchbuffer)(struct i965_batchbuffer_logger *logger,
                               const struct i965_logged_batchbuffer *from,
                               const struct i965_logged_batchbuffer *to);

   /**
    * To be called by the driver to add log-message data to the
    * batchbuffer log. The message will be added to the log of
    * the batchbuffer dst. If counter is non-NULL the values
    * in the counter are also emitted to the log.
    */
   void (*add_message)(struct i965_batchbuffer_logger *logger,
                       const struct i965_logged_batchbuffer *dst,
                       const char *txt);

   /**
    * call to release the i965_batchbuffer_logger
    */
   void (*release_driver)(struct i965_batchbuffer_logger *logger);

   /**
    * Create a counter object. If filename is non-NULL, then
    * the values of the counter will be emitted to the named
    * file when the counter is deleted.
    */
   struct i965_batchbuffer_counter(*create_counter)(struct i965_batchbuffer_logger *logger,
                                                    const char *filename);

   /**
    * Activate the counter, elements added to the active batchbuffer
    * of the calling thread will increment the counter values.
    */
   void (*activate_counter)(struct i965_batchbuffer_logger *logger,
                            struct i965_batchbuffer_counter counter);

   /**
    * Dectivate the counter, pausing counting operations.
    */
   void (*deactivate_counter)(struct i965_batchbuffer_logger *logger,
                              struct i965_batchbuffer_counter counter);

   /**
    * Record into the log counter values.
    */
   void (*print_counter)(struct i965_batchbuffer_logger *logger,
                         struct i965_batchbuffer_counter counter, const char *label);

   /**
    * Reset the counter values to zero.
    */
   void (*reset_counter)(struct i965_batchbuffer_logger *logger,
                         struct i965_batchbuffer_counter counter);

   /**
    * Release the counter object
    */
   void (*release_counter)(struct i965_batchbuffer_logger *logger,
                           struct i965_batchbuffer_counter counter);
};


/**
 * Function provided BY the 3D driver to return the offset into
 * the passed batch buffer of where the next command is to be
 * written.
 */
typedef uint32_t
(*i965_logged_batchbuffer_state)(const struct i965_logged_batchbuffer*);

/**
 * Function provided BY the 3D driver to tell i965_batchbuffer_logger
 * what batch buffer is active on the calling thread; if there is
 * no active batch buffer on the calling thread, then it sets
 * i965_logged_batchbuffer::fd as -1. For the case of
 * fd != -1, the value for i965_logged_batchbuffer::driver_data
 * is saved and associated to the (fd, gem_bo) key value until
 * the batch buffer is executed via drmIoctl or is aborted via
 * i965_logged_batchbuffer::aborted_batchbuffer().
 */
typedef void (*i965_active_batchbuffer)(struct i965_logged_batchbuffer*);

/**
 * The function pointer type for i965_batchbuffer_logger_acquire,
 * A 3D driver should use dlsym to find the symbol
 * i965_batchbuffer_logger_acquire and use it to acquire an
 * i965_batchbuffer_logger object.
 */
typedef struct i965_batchbuffer_logger*
(*i965_batchbuffer_logger_acquire_fcn)(int pci_id,
                                       i965_logged_batchbuffer_state f1,
                                       i965_active_batchbuffer f2);

#ifdef __cplusplus
}
#endif

#endif

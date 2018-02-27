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

#ifndef I965_BATCHBUFFER_LOGGER_OUTPUT_H
#define I965_BATCHBUFFER_LOGGER_OUTPUT_H

#include <stdint.h>
#include "i965_batchbuffer_logger_app.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * FILE format of the log is a seqence of (HEADER, NAME-DATA, VALUE-DATA)
 * tuples where HEADER is the struct i965_batchbuffer_logger_header packed
 * via fwrite, NAME-DATA is a sequence of HEADER.name_length characers and
 * VALUE-DATA is a seqnece of HEADER.value_length characters. The start
 * and end of blocks are used to give nested structure to the data (for
 * example to make better JSON or XML output). Block endings will have that
 * the name and value lengths are both ALWAYS 0. For other types, name should
 * never be zero, but value can be (typically for blocks).
 */

struct i965_batchbuffer_logger_header {
   enum i965_batchbuffer_logger_message_type_t type;

   /**
    * length of the string for the name must be 0 for type
    * I965_BATCHBUFFER_LOGGER_MESSAGE_BLOCK_END
    */
   uint32_t name_length;

   /**
    * length of the string for the value, must be 0 for types
    * I965_BATCHBUFFER_LOGGER_MESSAGE_BLOCK_BEGIN and
    * I965_BATCHBUFFER_LOGGER_MESSAGE_BLOCK_END
    */
   uint32_t value_length;
};

#ifdef __cplusplus
}
#endif

#endif

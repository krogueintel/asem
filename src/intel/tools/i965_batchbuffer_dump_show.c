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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#include "tools/i965_batchbuffer_logger_output.h"

static
void
print_tabs(int l)
{
   for(int i = 0; i < l; ++i) {
      printf("\t");
   }
}

static
void
print_instrumentation_message(int block_level,
                              const struct i965_batchbuffer_logger_header *hdr,
                              FILE *pfile)
{
   bool line_start;
   uint32_t i;
   char c = 0;
   size_t num_read;

   /* assume name has no EOLs */
   print_tabs(block_level);
   for (i = 0; !feof(pfile) && i < hdr->name_length; ++i) {
      num_read = fread(&c, sizeof(c), 1, pfile);
      if (num_read != 1)
         return;
      printf("%c", c);
   }

   if (hdr->value_length > 0) {
      printf(" : ");
   }

   /* print the value (if there is one). */
   for (i = 0, line_start = false; !feof(pfile) && i < hdr->value_length; ++i) {
      num_read = fread(&c, sizeof(c), 1, pfile);

      if (num_read != 1)
         break;

      if (line_start) {
         print_tabs(block_level);
      }

      printf("%c", c);
      line_start = (c == '\n');
   }

   if (c != '\n') {
      printf("\n");
   }
}


int
main(int argc, char **argv)
{
   FILE *pfile;
   int block_level = 0;

   if (argc != 2) {
      return -1;
   }

   pfile = fopen(argv[1], "r");
   if (pfile == NULL) {
      return -1;
   }

   while (!feof(pfile)) {
      struct i965_batchbuffer_logger_header hdr;
      size_t num_read;

      num_read = fread(&hdr, sizeof(hdr), 1, pfile);
      if (num_read != 1)
         break;

      switch(hdr.type) {
         case I965_BATCHBUFFER_LOGGER_MESSAGE_BLOCK_BEGIN: {
            print_instrumentation_message(block_level, &hdr, pfile);
            ++block_level;
         }
         break;

         case I965_BATCHBUFFER_LOGGER_MESSAGE_BLOCK_END: {
            --block_level;
            if (block_level < 0) {
               fprintf(stderr, "Warning: negative block level encountered\n");
            }
         }
         break;

         case I965_BATCHBUFFER_LOGGER_MESSAGE_VALUE: {
            print_instrumentation_message(block_level, &hdr, pfile);
         }
         break;
      }
   }

   if (block_level != 0) {
      fprintf(stderr, "Warning: did not end on positive block level\n");
   }
   fclose(pfile);
   return 0;
}

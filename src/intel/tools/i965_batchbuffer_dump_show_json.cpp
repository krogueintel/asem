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

#include <sstream>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <iterator>
#include <cstdio>
#include <cstdlib>
#include <assert.h>

#include "tools/i965_batchbuffer_logger_output.h"

static
std::string
json_name_from_file(unsigned int len, std::FILE *pfile)
{
   std::string return_value;
   unsigned int i;
   char c;

   for(i = 0; i < len; ++i) {
      if (std::fread(&c, sizeof(char), 1, pfile) != sizeof(char)) {
         break;
      }
      if(c != '\n') {
         return_value.push_back(c);
      }
   }

   return return_value;
}

static
std::string
json_block_name_from_file(const struct i965_batchbuffer_logger_header *hdr,
                          std::FILE *pfile)
{
   std::ostringstream return_value;

   return_value << json_name_from_file(hdr->name_length, pfile);
   if (hdr->value_length > 0) {
      return_value << ":" << json_name_from_file(hdr->value_length, pfile);
   }

   return return_value.str();
}

static
void
json_print_value_line(std::vector<char>::const_iterator begin,
                      std::vector<char>::const_iterator end)
{
   for(; begin != end; ++begin) {
      switch(*begin) {
      case '\t':
         std::cout << "\\t";
         break;
      case '\"':
         std::cout << "\\\"";
         break;
      case '\\':
         /* This is silly but required; apirtace will place control
          * character codes (sometimes) within string values (typically
          * from shader sources). So for example if a shader source
          * has (within a comment) something like "Famous"
          * then the detailed function value will have then \"Famour\"
          * within its string value. The below silly block of code
          * just checks if there is a non-white character and if so,
          * just assume that the \ is a control code.
          */
         if (begin != end) {
            ++begin;
            if(isspace(*begin)) {
               std::cout << "\\";
            } else {
               std::cout << "\\" << *begin;
            }
         } else {
            std::cout << "\\";
         }
         break;
      default:
         std::cout << *begin;
      }
   }
}

static
void
json_print_value_from_file(unsigned int len, std::FILE *pfile)
{
   std::vector<char> value(len);
   std::vector<char>::iterator iter;
   std::vector<char>::iterator prev_iter;

   len = std::fread(&value[0], sizeof(char), len, pfile);
   value.resize(len);

   while(!value.empty() && value.back() == '\n') {
      value.pop_back();
   }

   for (prev_iter = value.begin(); prev_iter != value.end() && *prev_iter == '\n'; ++prev_iter)
   {}

   iter = std::find(prev_iter, value.end(), '\n');
   if (iter == value.end()) {
      std::cout << "\"";
      json_print_value_line(prev_iter, value.end());
      std::cout << "\"\n";
   } else {
      unsigned int line(0);

      std::cout << "{";

      while(iter != value.end()) {
         std::cout << "\"line-" << std::setw(5) << line << "\":\"";
         json_print_value_line(prev_iter, iter);
         std::cout << "\",\n";

         ++iter;
         prev_iter = iter;
         iter = std::find(prev_iter, value.end(), '\n');
         ++line;
      }
      std::cout << "\"line-" << std::setw(5) << line << "\":\"";
      json_print_value_line(prev_iter, iter);
      std::cout << "\"\n";
      std::cout << "}\n";
   }
}

static
void
handle_block_begin(const struct i965_batchbuffer_logger_header *hdr,
                   std::vector<unsigned int> &block_stack,
                   std::FILE *pfile)
{
   std::string name;

   assert(hdr->type == I965_BATCHBUFFER_LOGGER_MESSAGE_BLOCK_BEGIN);

   if(block_stack.back() != 0) {
      std::cout << ",\n";
   }

   block_stack.push_back(0);
   name = json_block_name_from_file(hdr, pfile);
   std::cout << "\"" << name << "\":{\n";
}

static
void
handle_block_end(const struct i965_batchbuffer_logger_header *hdr,
                 std::vector<unsigned int> &block_stack,
                 std::FILE *pfile)
{
   if (block_stack.size() > 1) {
      std::cout << "}";
      block_stack.pop_back();
      ++block_stack.back();
   }
}

static
void
handle_message(const struct i965_batchbuffer_logger_header *hdr,
               std::vector<unsigned int> &block_stack,
               std::FILE *pfile)
{
   std::string name;

   if(block_stack.back() != 0) {
      std::cout << ",\n";
   }

   name = json_name_from_file(hdr->name_length, pfile);
   std::cout << "\"" << name << "\":";
   json_print_value_from_file(hdr->value_length, pfile);
   ++block_stack.back();
}

int
main(int argc, char **argv)
{
   std::vector<unsigned int> block_stack;
   std::FILE *pfile;

   if (argc != 2) {
      return -1;
   }

   pfile = std::fopen(argv[1], "r");
   if (!pfile) {
      return -1;
   }

   block_stack.push_back(0);

   std::cout << "{\n";
   while (!feof(pfile)) {
      struct i965_batchbuffer_logger_header hdr;
      size_t num_read;

      num_read = fread(&hdr, sizeof(hdr), 1, pfile);
      if (num_read != 1)
         break;

      switch(hdr.type) {
         case I965_BATCHBUFFER_LOGGER_MESSAGE_BLOCK_BEGIN: {
            handle_block_begin(&hdr, block_stack, pfile);
         }
         break;

         case I965_BATCHBUFFER_LOGGER_MESSAGE_BLOCK_END: {
            handle_block_end(&hdr, block_stack, pfile);
         }
         break;

         case I965_BATCHBUFFER_LOGGER_MESSAGE_VALUE: {
            handle_message(&hdr, block_stack, pfile);
         }
         break;
      }
   }

   std::cout << "}\n";
   fclose(pfile);
   return 0;
}

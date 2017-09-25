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

#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <assert.h>

#include "tools/i965_batchbuffer_logger_output.h"

static
std::string
xml_value_from_file(unsigned int len, std::FILE *pfile)
{
   std::ostringstream str;
   unsigned int i;
   char c;

   str << "<![CDATA[";
   for(i = 0; i < len; ++i) {
      if (std::fread(&c, sizeof(char), 1, pfile) != sizeof(char)) {
         break;
      }
      str << c;
   }
   str << "]]>";
   return str.str();
}

static
std::string
xml_string_from_file(unsigned int len, std::FILE *pfile)
{
   std::ostringstream str;
   unsigned int i;
   char c;

   for(i = 0; i < len; ++i) {
      if (std::fread(&c, sizeof(char), 1, pfile) != sizeof(char)) {
         break;
      }
      switch(c) {
      case '<':
         str << "&lt;";
         break;
      case '>':
         str << "&gt;";
         break;
      case '&':
         str << "&amp;";
         break;
      case '\n':
         str << "&#13;&#10;";
         break;
      case '\"':
         str << "&quot;";
         break;
      default:
         str << c;
      }
   }

   return str.str();
}

static
bool
legal_tag_char(char c)
{
   switch(c)
   {
   case '-':
   case '_':
   case '.':
      return true;
   }
   return !std::isspace(c) && std::isalnum(c);
}

static
std::string
xml_tag_from_file(unsigned int len, std::FILE *pfile)
{
   std::string return_value;
   unsigned int i;
   char c;

   for(i = 0; i < len; ++i) {
      if (std::fread(&c, sizeof(char), 1, pfile) != sizeof(char)) {
         break;
      }
      if (legal_tag_char(c)) {
         if (i == 0 && (std::isdigit(c) || c == '.' || c == '-')) {
            return_value.push_back('_');
         }
         return_value.push_back(c);
      } else {
         return_value.push_back('_');
      }
   }

   return return_value;
}

static
void
handle_block_begin(const struct i965_batchbuffer_logger_header *hdr,
                   std::vector<std::string> &block_stack,
                   std::FILE *pfile)
{
   assert(hdr->type == I965_BATCHBUFFER_LOGGER_MESSAGE_BLOCK_BEGIN);
   block_stack.push_back(xml_tag_from_file(hdr->name_length, pfile));
   std::cout << "<" << block_stack.back();
   if (hdr->value_length > 0) {
      std::cout << " value=\"" << xml_string_from_file(hdr->value_length, pfile)
                << "\"";
   }
   std::cout << ">\n";
}

static
void
handle_block_end(const struct i965_batchbuffer_logger_header *hdr,
                 std::vector<std::string> &block_stack,
                 std::FILE *pfile)
{
   if (!block_stack.empty()) {
      std::cout << "</" << block_stack.back()
                << ">\n";
      block_stack.pop_back();
   }
}

static
void
handle_message(const struct i965_batchbuffer_logger_header *hdr,
               std::FILE *pfile)
{
   std::string name;
   name = xml_tag_from_file(hdr->name_length, pfile);
   std::cout << "<" << name << ">\n";
   if (hdr->value_length > 0) {
      std::cout << xml_value_from_file(hdr->value_length, pfile);
   }
   std::cout << "</" << name << ">\n";
}

int
main(int argc, char **argv)
{
   std::vector<std::string> block_stack;
   std::FILE *pfile;

   if (argc != 2) {
      return -1;
   }

   pfile = std::fopen(argv[1], "r");
   if (!pfile) {
      return -1;
   }

   std::cout << "<trace>\n";
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
            handle_message(&hdr, pfile);
         }
         break;
      }
   }

   std::cout << "</trace>\n";
   fclose(pfile);
   return 0;
}

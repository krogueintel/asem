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
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <getopt.h>
#include <strings.h>

#include "compiler/brw_inst.h"
#include "compiler/brw_eu.h"

static void
print_opcodes(const void *data, int data_sz,
              struct gen_device_info *devinfo,
              bool print_offsets)
{
   for (int offset = 0; offset < data_sz;) {
      const brw_inst *insn = data + offset;
      bool compacted;
      brw_inst uncompacted;
      enum opcode opcode;
      const struct opcode_desc *desc;

      if (print_offsets) {
         printf("0x%08x: ", offset);
      }

      compacted = brw_inst_cmpt_control(devinfo, insn);
      if (compacted) {
         brw_compact_inst *compacted_insn = (brw_compact_inst *)insn;
         brw_uncompact_instruction(devinfo, &uncompacted, compacted_insn);
         insn = &uncompacted;
         offset += 8;
      } else {
         offset += 16;
      }

      opcode = brw_inst_opcode(devinfo, insn);
      desc = brw_opcode_desc(devinfo, opcode);
      if (desc) {
         printf("(0x%08x) %s", opcode, desc->name);
      } else {
         printf("(0x%08x) UnknownOpcode", opcode);
      }

      if (compacted) {
         printf(" {compacted}");
      }

      printf("\n");
   }
}

static void
print_disassembly(const void *data, int data_sz,
                  struct gen_device_info *devinfo,
                  bool print_offsets)
{
   struct disasm_info *disasm_info = disasm_initialize(devinfo, NULL);
   disasm_new_inst_group(disasm_info, 0);
   disasm_new_inst_group(disasm_info, data_sz);

   brw_validate_instructions(devinfo, data, 0, data_sz, disasm_info);

   foreach_list_typed_safe(struct inst_group, group, link,
                           &disasm_info->group_list) {
      if (exec_node_is_tail_sentinel(exec_node_get_next(&group->link)))
         break;

      struct inst_group *next =
         exec_node_data(struct inst_group,
                        exec_node_get_next(&group->link), link);

      int start_offset = group->offset;
      int end_offset = next->offset;

      brw_disassemble(devinfo, data, start_offset,
                      end_offset, print_offsets, stdout);

      if (group->error) {
         fputs(group->error, stdout);
      }
   }

   ralloc_free(disasm_info);
}

static
void
print_help(FILE *p, const char *argv0)
{
   fprintf(p, "Usage: %s [OPTION] ShaderBinary\n"
           "  -p   PCI ID of GPU for which shader binary was generated, must be given;\n"
           "       if the value is of the form 0xVALUE, then read the value as hex;\n"
           "       if the value is of the form 0VALUE, then read the value as oct;\n"
           "       otherwise read the value as decimal\n"
           "  -c   Print opcodes only instead of disassembly\n"
           "  -o   Print offsets into ShaderBinary\n"
           "  -h   Print this help message and quit\n", argv0);
}

int
main(int argc, char **argv)
{
   const char *short_options = "p:och";
   FILE *file;
   void *data;
   int data_sz, pci_id, data_read;
   bool pci_id_provided = false, print_opcodes_only = false;
   bool print_offsets = false;
   struct gen_device_info devinfo;

   int opt;
   while ((opt = getopt(argc, argv, short_options)) != -1) {
      switch(opt) {
      case 'p':
         pci_id_provided = true;
         pci_id = strtol(optarg, NULL, 0);
         break;
      case 'c':
         print_opcodes_only = true;
         break;
      case 'o':
         print_offsets = true;
         break;
      case 'h':
         print_help(stdout, argv[0]);
         return 0;
      default:
         fprintf(stderr, "Unknown option '%c'\n", (char)opt);
         print_help(stderr, argv[0]);
         return -1;
      }
   }

   if (optind >= argc) {
      fprintf(stderr, "Need to proviade ShaderBinary file.\n");
      print_help(stderr, argv[0]);
      return -1;
   }

   if (!pci_id_provided) {
      fprintf(stderr, "Need to provide PCI ID with -p option\n");
      return -1;
   }

   if(!gen_get_device_info(pci_id, &devinfo)) {
      fprintf(stderr, "Bad PCIID 0x%x given, aborting\n", pci_id);
      return -1;
   }

   file = fopen(argv[optind], "r");
   if (file == NULL) {
      fprintf(stderr, "Unable to open file \"%s\" for reading, aborting\n",
              argv[optind]);
      return -1;
   }

   fseek(file, 0, SEEK_END);
   data_sz = ftell(file);
   fseek(file, 0, SEEK_SET);

   data = malloc(data_sz);
   if (data == NULL) {
      fprintf(stderr,
              "Failed to allocate %d bytes to hold file contents.\n",
              data_sz);
      return -1;
   }

   data_read = fread(data, 1, data_sz, file);
   if (data_read != data_sz) {
      fprintf(stderr,
              "Failed to read entire file \"%s\", read %d of %d bytes\n",
              argv[optind], data_read, data_sz);
      free(data);
      fclose(file);
      return -1;
   }

   brw_init_compaction_tables(&devinfo);
   if (print_opcodes_only) {
      print_opcodes(data, data_sz, &devinfo, print_offsets);
   } else {
      print_disassembly(data, data_sz, &devinfo, print_offsets);
   }

   free(data);
   fclose(file);
   return 0;
}

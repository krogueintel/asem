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

#ifndef _i965_INSTRUMENTATION_INSTRUCTIONS_H_
#define _i965_INSTRUMENTATION_INSTRUCTIONS_H_

#define STATE_BASE_ADDRESS                  0x61010000

#define MEDIA_INTERFACE_DESCRIPTOR_LOAD     0x70020000
#define MEDIA_CURBE_LOAD                    0x70010000
#define MEDIA_VFE_STATE                     0x70000000
#define MEDIA_STATE_FLUSH                   0x70040000

#define _3DSTATE_PIPELINE_SELECT            0x61040000
#define _3DSTATE_PIPELINE_SELECT_GM45       0x69040000

#define _3DSTATE_INDEX_BUFFER               0x780a0000
#define _3DSTATE_VERTEX_BUFFERS             0x78080000

#define _3DSTATE_VF_INSTANCING              0x78490000

#define _3DSTATE_VS                         0x78100000
#define _3DSTATE_GS                         0x78110000
#define _3DSTATE_HS                         0x781b0000
#define _3DSTATE_DS                         0x781d0000
#define _3DSTATE_PS                         0x78200000

#define _3D_STATE_CLIP                      0x78120000

#define _3DSTATE_CONSTANT_VS                0x78150000
#define _3DSTATE_CONSTANT_GS                0x78160000
#define _3DSTATE_CONSTANT_PS                0x78170000
#define _3DSTATE_CONSTANT_HS                0x78190000
#define _3DSTATE_CONSTANT_DS                0x781A0000

#define _3DSTATE_BINDING_TABLE_POINTERS_VS  0x78260000
#define _3DSTATE_BINDING_TABLE_POINTERS_HS  0x78270000
#define _3DSTATE_BINDING_TABLE_POINTERS_DS  0x78280000
#define _3DSTATE_BINDING_TABLE_POINTERS_GS  0x78290000
#define _3DSTATE_BINDING_TABLE_POINTERS_PS  0x782a0000

#define _3DSTATE_SAMPLER_STATE_POINTERS_VS  0x782b0000
#define _3DSTATE_SAMPLER_STATE_POINTERS_DS  0x782c0000
#define _3DSTATE_SAMPLER_STATE_POINTERS_HS  0x782d0000
#define _3DSTATE_SAMPLER_STATE_POINTERS_GS  0x782e0000
#define _3DSTATE_SAMPLER_STATE_POINTERS_PS  0x782f0000
#define _3DSTATE_SAMPLER_STATE_POINTERS     0x78020000

#define _3DSTATE_VIEWPORT_STATE_POINTERS_CC 0x78230000
#define _3DSTATE_VIEWPORT_STATE_POINTERS_SF_CLIP 0x78210000
#define _3DSTATE_BLEND_STATE_POINTERS       0x78240000
#define _3DSTATE_CC_STATE_POINTERS          0x780e0000
#define _3DSTATE_SCISSOR_STATE_POINTERS     0x780f0000

#define _MI_CMD_3D                          (0x3 << 29)
#define _3DSTATE_PIPE_CONTROL               (_MI_CMD_3D | (3 << 27) | (2 << 24))

#define _3DPRIMITIVE                        0x7b000000
#define _GPGPU_WALKER                       0x71050000

#define _MI_CMD                             (0x0 << 29)

/* _MI's that set values of registers that we can (mostly)
 * determine the value after the kernel returns the ioctl.
 */
#define _MI_LOAD_REGISTER_IMM		    (_MI_CMD | (34 << 23))
#define _MI_LOAD_REGISTER_REG		    (_MI_CMD | (42 << 23))
#define _MI_LOAD_REGISTER_MEM               (_MI_CMD | (41 << 23))
#define _MI_STORE_REGISTER_MEM              (_MI_CMD | (36 << 23))

/* _MI_'s that are commands, not all of these are allowed
 * in an execlist
 */
#define _MI_NOOP                            (_MI_CMD | ( 0 << 23))
#define _MI_BATCH_BUFFER_END                (_MI_CMD | (10 << 23))
#define _MI_BATCH_BUFFER_START              (_MI_CMD | (49 << 23))
#define _MI_ARB_CHECK                       (_MI_CMD | ( 5 << 23))
#define _MI_ATOMIC                          (_MI_CMD | (47 << 23))
#define _MI_CLFLUSH                         (_MI_CMD | (39 << 23))
#define _MI_CONDITIONAL_BATCH_BUFFER_END    (_MI_CMD | (54 << 23))
#define _MI_COPY_MEM_MEM                    (_MI_CMD | (46 << 23))
#define _MI_DISPLAY_FLIP                    (_MI_CMD | (20 << 23))
#define _MI_FORCE_WAKEUP                    (_MI_CMD | (29 << 23))
#define _MI_LOAD_SCAN_LINES_EXCL            (_MI_CMD | (19 << 23))
#define _MI_LOAD_SCAN_LINES_INCL            (_MI_CMD | (18 << 23))
#define _MI_MATH                            (_MI_CMD | (26 << 23))
#define _MI_REPORT_HEAD                     (_MI_CMD | ( 7 << 23))
#define _MI_REPORT_PERF_COUNT               (_MI_CMD | (40 << 23))
#define _MI_RS_CONTEXT                      (_MI_CMD | (15 << 23))
#define _MI_RS_CONTROL                      (_MI_CMD | ( 6 << 23))
#define _MI_RS_STORE_DATA_IMM               (_MI_CMD | (43 << 23))
#define _MI_SEMAPHORE_SIGNAL                (_MI_CMD | (27 << 23))
#define _MI_SEMAPHORE_WAIT                  (_MI_CMD | (28 << 23))
#define _MI_SET_CONTEXT                     (_MI_CMD | (24 << 23))
#define _MI_STORE_DATA_IMM                  (_MI_CMD | (32 << 23))
#define _MI_STORE_DATA_INDEX                (_MI_CMD | (33 << 23))
#define _MI_SUSPEND_FLUSH                   (_MI_CMD | (11 << 23))
#define _MI_UPDATE_GTT                      (_MI_CMD | (35 << 23))
#define _MI_USER_INTERRUPT                  (_MI_CMD | ( 2 << 23))
#define _MI_WAIT_FOR_EVENT                  (_MI_CMD | ( 3 << 23))
/* setting the predicate directly or via registers is viewed
 * as a command and not state because the value to which it is set
 * is not entirely determined by CPU values.
 */
#define _MI_SET_PREDICATE                   (_MI_CMD | ( 1 << 23))
#define _MI_PREDICATE                       (_MI_CMD | (12 << 23))

/* _MI_'s that set state value */
#define _MI_TOPOLOGY_FILTER                 (_MI_CMD | 13 << 23)

#endif

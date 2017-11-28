#include "brw_context.h"
#include "brw_defines.h"
#include "intel_mipmap_tree.h"

void
gen9_set_astc5x5_wa_mode(struct brw_context *brw,
                         enum brw_astc5x5_wa_mode_t mode)
{
   if (!brw->astc5x5_wa.required ||
       mode == BRW_ASTC5x5_WA_MODE_NONE ||
       brw->astc5x5_wa.mode == mode) {
      return;
   }

   if (brw->astc5x5_wa.mode != BRW_ASTC5x5_WA_MODE_NONE) {
      const uint32_t flags = PIPE_CONTROL_CS_STALL |
         PIPE_CONTROL_TEXTURE_CACHE_INVALIDATE;
      brw_emit_pipe_control_flush(brw, flags);
   }

   brw->astc5x5_wa.mode = mode;
}

void
gen9_astc5x5_perform_wa(struct brw_context *brw)
{
   if (!brw->astc5x5_wa.required) {
      return;
   }

   if (brw->astc5x5_wa.texture_astc5x5_present) {
      gen9_set_astc5x5_wa_mode(brw, BRW_ASTC5x5_WA_MODE_HAS_ASTC5x5);
   } else if (brw->astc5x5_wa.texture_with_auxilary_present) {
      gen9_set_astc5x5_wa_mode(brw, BRW_ASTC5x5_WA_MODE_HAS_AUX);
   }
}

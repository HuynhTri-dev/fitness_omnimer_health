import { COLORS } from './colors';
import { RADIUS } from './radius';
import { SPACING } from './spacing';
import { TYPOGRAPHY } from './typography';

export const darkTheme = {
  COLORS: {
    ...COLORS,
    BACKGROUND: COLORS.GRAY_900,
    TEXT_PRIMARY: COLORS.WHITE,
    SURFACE: COLORS.GRAY_800,
  },
  TYPOGRAPHY,
  SPACING,
  RADIUS,
};

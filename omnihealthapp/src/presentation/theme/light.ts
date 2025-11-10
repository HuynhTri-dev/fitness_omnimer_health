import { COLORS } from './colors';
import { RADIUS } from './radius';
import { SPACING } from './spacing';
import { TYPOGRAPHY } from './typography';

export const lightTheme = {
  COLORS: {
    ...COLORS,
    BACKGROUND: COLORS.GRAY_100,
    TEXT_PRIMARY: COLORS.TEXT_PRIMARY,
    SURFACE: COLORS.WHITE,
  },
  TYPOGRAPHY,
  SPACING,
  RADIUS,
};

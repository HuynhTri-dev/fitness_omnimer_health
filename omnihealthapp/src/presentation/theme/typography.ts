// src/theme/typography.ts

export interface ITypography {
  fontFamily: {
    headingBold: string;
    headingRegular: string;
    bodyBold: string;
    bodyRegular: string;
    bodyItalic: string;
  };
  fontSize: {
    xs: number;
    sm: number;
    base: number;
    lg: number;
    xl: number;
    '2xl': number;
  };
  lineHeight: {
    tight: number;
    normal: number;
    relaxed: number;
  };
}

// Typography constant
export const TYPOGRAPHY: ITypography = {
  fontFamily: {
    headingBold: 'Orbitron-Bold',
    headingRegular: 'Orbitron-Regular',
    bodyBold: 'Montserrat-Bold',
    bodyRegular: 'Montserrat-Regular',
    bodyItalic: 'Montserrat-Italic',
  },
  fontSize: {
    xs: 12,
    sm: 14,
    base: 16,
    lg: 20,
    xl: 24,
    '2xl': 32,
  },
  lineHeight: {
    tight: 1.2,
    normal: 1.5,
    relaxed: 1.8,
  },
};

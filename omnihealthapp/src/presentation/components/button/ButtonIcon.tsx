import { COLORS, RADIUS, TYPOGRAPHY } from '@/presentation/theme';

import React from 'react';
import {
  TouchableOpacity,
  Text,
  StyleSheet,
  ActivityIndicator,
  ViewStyle,
  TextStyle,
  View,
} from 'react-native';

export enum ButtonVariant {
  PrimarySolid = 'PrimarySolid',
  SecondarySolid = 'SecondarySolid',
  DangerSolid = 'DangerSolid',
  PrimaryOutline = 'PrimaryOutline',
  SecondaryOutline = 'SecondaryOutline',
  DangerOutline = 'DangerOutline',
}

interface ButtonIconProps {
  variant?: ButtonVariant;
  title?: string;
  icon: React.ReactNode;
  disabled?: boolean;
  loading?: boolean;
  onPress?: () => void;
  fontSize?: number;
  fontWeight?: 'bold' | 'regular';
  style?: ViewStyle;
}

export const ButtonIcon: React.FC<ButtonIconProps> = ({
  variant = ButtonVariant.PrimarySolid,
  title,
  icon,
  disabled = false,
  loading = false,
  onPress,
  fontSize = TYPOGRAPHY.fontSize.base,
  fontWeight = 'bold',
  style,
}) => {
  const [isPressed, setIsPressed] = React.useState(false);

  const getBackgroundColor = (): string => {
    if (disabled || loading) {
      return COLORS.GRAY_400;
    }

    const isSolid = variant.includes('Solid');

    if (!isSolid) {
      return 'transparent';
    }

    if (isPressed) {
      switch (variant) {
        case ButtonVariant.PrimarySolid:
          return COLORS.SECONDARY;
        case ButtonVariant.SecondarySolid:
          return COLORS.GRAY_800;
        case ButtonVariant.DangerSolid:
          return COLORS.DANGER_HOVER;
        default:
          return COLORS.PRIMARY;
      }
    }

    switch (variant) {
      case ButtonVariant.PrimarySolid:
        return COLORS.PRIMARY;
      case ButtonVariant.SecondarySolid:
        return COLORS.GRAY_600;
      case ButtonVariant.DangerSolid:
        return COLORS.DANGER;
      default:
        return COLORS.PRIMARY;
    }
  };

  const getBorderColor = (): string => {
    if (disabled || loading) {
      return COLORS.GRAY_400;
    }

    const isOutline = variant.includes('Outline');
    if (!isOutline) return 'transparent';

    if (isPressed) {
      switch (variant) {
        case ButtonVariant.PrimaryOutline:
          return COLORS.SECONDARY;
        case ButtonVariant.SecondaryOutline:
          return COLORS.GRAY_800;
        case ButtonVariant.DangerOutline:
          return COLORS.DANGER_HOVER;
        default:
          return COLORS.PRIMARY;
      }
    }

    switch (variant) {
      case ButtonVariant.PrimaryOutline:
        return COLORS.PRIMARY;
      case ButtonVariant.SecondaryOutline:
        return COLORS.GRAY_600;
      case ButtonVariant.DangerOutline:
        return COLORS.DANGER;
      default:
        return COLORS.PRIMARY;
    }
  };

  const getTextColor = (): string => {
    if (disabled || loading) {
      return COLORS.TEXT_MUTED;
    }

    const isOutline = variant.includes('Outline');
    if (isOutline) {
      if (isPressed) {
        switch (variant) {
          case ButtonVariant.PrimaryOutline:
            return COLORS.SECONDARY;
          case ButtonVariant.SecondaryOutline:
            return COLORS.GRAY_800;
          case ButtonVariant.DangerOutline:
            return COLORS.DANGER_HOVER;
          default:
            return COLORS.PRIMARY;
        }
      }

      switch (variant) {
        case ButtonVariant.PrimaryOutline:
          return COLORS.PRIMARY;
        case ButtonVariant.SecondaryOutline:
          return COLORS.GRAY_600;
        case ButtonVariant.DangerOutline:
          return COLORS.DANGER;
        default:
          return COLORS.PRIMARY;
      }
    }

    return COLORS.WHITE;
  };

  const getFontFamily = (): string => {
    return fontWeight === 'bold'
      ? TYPOGRAPHY.fontFamily.bodyBold
      : TYPOGRAPHY.fontFamily.bodyRegular;
  };

  const buttonStyle: ViewStyle = {
    ...styles.button,
    backgroundColor: getBackgroundColor(),
    borderColor: getBorderColor(),
    borderWidth: variant.includes('Outline') ? 2 : 0,
    opacity: disabled && !loading ? 0.5 : 1,
    ...style,
  };

  const textStyle: TextStyle = {
    ...styles.text,
    color: getTextColor(),
    fontSize,
    fontFamily: getFontFamily(),
  };

  return (
    <TouchableOpacity
      style={buttonStyle}
      onPress={onPress}
      disabled={disabled || loading}
      onPressIn={() => setIsPressed(true)}
      onPressOut={() => setIsPressed(false)}
      activeOpacity={0.9}
    >
      {loading ? (
        <ActivityIndicator color={getTextColor()} size="small" />
      ) : (
        <View style={styles.content}>
          <View style={styles.iconContainer}>{icon}</View>
          {title && <Text style={textStyle}>{title}</Text>}
        </View>
      )}
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  button: {
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: RADIUS.md,
    alignItems: 'center',
    justifyContent: 'center',
    alignSelf: 'flex-start',
  },
  content: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  iconContainer: {
    marginRight: 0,
  },
  text: {
    textAlign: 'center',
    marginLeft: 8,
  },
});

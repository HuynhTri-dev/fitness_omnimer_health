/**
 * TextField Component - Enhanced Design
 *
 * Component input field với thiết kế năng động và chuyên nghiệp
 *
 * @example
 * Sử dụng cơ bản
 * <TextField
 *   label="Họ tên"
 *   value={name}
 *   onChangeText={setName}
 *   placeholder="Nhập họ tên của bạn"
 * />
 *
 * @example
 * Với icon và validation
 * <TextField
 *   label="Email"
 *   value={email}
 *   onChangeText={setEmail}
 *   leftIcon={<EmailIcon />}
 *   keyboardType="email-address"
 *   validators={[
 *     {
 *       validate: (value) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value),
 *       message: 'Email không hợp lệ'
 *     }
 *   ]}
 * />
 */

import { COLORS, RADIUS, SPACING, TYPOGRAPHY } from '@/presentation/theme';
import React, { useState, useRef } from 'react';
import {
  View,
  TextInput,
  Text,
  StyleSheet,
  ViewStyle,
  TextStyle,
  KeyboardTypeOptions,
  Animated,
} from 'react-native';

export enum TextFieldVariant {
  Primary = 'Primary',
  Secondary = 'Secondary',
}

export interface ValidationRule {
  validate: (value: string) => boolean;
  message: string;
}

interface TextFieldProps {
  value?: string;
  onChangeText?: (text: string) => void;
  keyboardType?: KeyboardTypeOptions;
  secureTextEntry?: boolean;
  editable?: boolean;
  label?: string;
  error?: string;
  helperText?: string;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  multiline?: boolean;
  numberOfLines?: number;
  maxLength?: number;
  variant?: TextFieldVariant;
  autoFocus?: boolean;
  validators?: ValidationRule[];
  placeholder?: string;
  style?: ViewStyle;
}

export const TextField: React.FC<TextFieldProps> = ({
  value = '',
  onChangeText,
  keyboardType = 'default',
  secureTextEntry = false,
  editable = true,
  label,
  error,
  helperText,
  leftIcon,
  rightIcon,
  multiline = false,
  numberOfLines = 1,
  maxLength,
  variant = TextFieldVariant.Primary,
  autoFocus = false,
  validators = [],
  placeholder,
  style,
}) => {
  const [isFocused, setIsFocused] = useState(false);
  const [internalError, setInternalError] = useState<string>('');
  const focusAnim = useRef(new Animated.Value(0)).current;
  const labelAnim = useRef(new Animated.Value(value ? 1 : 0)).current;

  const handleChangeText = (text: string) => {
    setInternalError('');

    // Animate label khi có/không có text
    Animated.timing(labelAnim, {
      toValue: text ? 1 : 0,
      duration: 200,
      useNativeDriver: false,
    }).start();

    if (validators.length > 0) {
      for (const rule of validators) {
        if (!rule.validate(text)) {
          setInternalError(rule.message);
          break;
        }
      }
    }

    onChangeText?.(text);
  };

  const handleFocus = () => {
    setIsFocused(true);
    Animated.parallel([
      Animated.timing(focusAnim, {
        toValue: 1,
        duration: 200,
        useNativeDriver: false,
      }),
      Animated.timing(labelAnim, {
        toValue: 1,
        duration: 200,
        useNativeDriver: false,
      }),
    ]).start();
  };

  const handleBlur = () => {
    setIsFocused(false);
    Animated.timing(focusAnim, {
      toValue: 0,
      duration: 200,
      useNativeDriver: false,
    }).start();

    if (!value) {
      Animated.timing(labelAnim, {
        toValue: 0,
        duration: 200,
        useNativeDriver: false,
      }).start();
    }

    if (validators.length > 0) {
      for (const rule of validators) {
        if (!rule.validate(value)) {
          setInternalError(rule.message);
          break;
        }
      }
    }
  };

  const getBorderColor = (): string => {
    if (!editable) return COLORS.GRAY_300;
    if (error || internalError) return COLORS.ERROR;

    if (variant === TextFieldVariant.Secondary) {
      if (isFocused) return COLORS.GRAY_700;
      if (value) return COLORS.GRAY_600;
      return COLORS.GRAY_400;
    }

    if (isFocused) return COLORS.SECONDARY;
    if (value) return COLORS.PRIMARY;
    return COLORS.GRAY_300;
  };

  const getIconBackgroundColor = (): string => {
    if (!editable) return COLORS.GRAY_100;
    if (error || internalError) return COLORS.ERROR + '15'; // 15% opacity

    if (variant === TextFieldVariant.Secondary) {
      if (isFocused || value) return COLORS.GRAY_100;
      return COLORS.GRAY_200;
    }

    if (isFocused) return COLORS.SECONDARY + '15';
    if (value) return COLORS.PRIMARY + '15';
    return COLORS.GRAY_200;
  };

  const displayError = error || internalError;

  const borderWidth = focusAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [1.5, 2.5],
  });

  const containerStyle: ViewStyle = {
    ...styles.container,
    ...style,
  };

  const inputContainerStyle = {
    ...styles.inputContainer,
    borderColor: getBorderColor(),
    opacity: !editable ? 0.5 : 1,
    minHeight: multiline ? numberOfLines * 24 + 24 : 56,
  };

  const animatedBorderStyle = {
    borderWidth,
  };

  const inputStyle: TextStyle = {
    ...styles.input,
    fontFamily: TYPOGRAPHY.fontFamily.bodyRegular,
    fontSize: TYPOGRAPHY.fontSize.base,
    textAlignVertical: multiline ? 'top' : 'center',
  };

  const labelStyle: TextStyle = {
    ...styles.label,
    fontFamily: TYPOGRAPHY.fontFamily.bodyBold,
    fontSize: TYPOGRAPHY.fontSize.sm,
    color: displayError
      ? COLORS.ERROR
      : isFocused
      ? variant === TextFieldVariant.Secondary
        ? COLORS.GRAY_700
        : COLORS.SECONDARY
      : COLORS.TEXT_PRIMARY,
  };

  const errorStyle: TextStyle = {
    ...styles.helperText,
    fontFamily: TYPOGRAPHY.fontFamily.bodyRegular,
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: COLORS.ERROR,
  };

  const helperStyle: TextStyle = {
    ...styles.helperText,
    fontFamily: TYPOGRAPHY.fontFamily.bodyRegular,
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: COLORS.TEXT_SECONDARY,
  };

  const iconContainerStyle: ViewStyle = {
    ...styles.iconContainer,
    backgroundColor: getIconBackgroundColor(),
  };

  return (
    <View style={containerStyle}>
      {label && <Text style={labelStyle}>{label}</Text>}

      <Animated.View style={[inputContainerStyle, animatedBorderStyle]}>
        {leftIcon && (
          <View style={[iconContainerStyle, styles.leftIconContainer]}>
            {leftIcon}
          </View>
        )}

        <TextInput
          value={value}
          onChangeText={handleChangeText}
          onFocus={handleFocus}
          onBlur={handleBlur}
          keyboardType={keyboardType}
          secureTextEntry={secureTextEntry}
          editable={editable}
          multiline={multiline}
          numberOfLines={numberOfLines}
          maxLength={maxLength}
          autoFocus={autoFocus}
          placeholder={placeholder}
          placeholderTextColor={COLORS.TEXT_MUTED}
          style={inputStyle}
        />

        {rightIcon && (
          <View style={[iconContainerStyle, styles.rightIconContainer]}>
            {rightIcon}
          </View>
        )}

        {maxLength && (
          <Text style={styles.charCount}>
            {value.length}/{maxLength}
          </Text>
        )}
      </Animated.View>

      {displayError ? (
        <View style={styles.errorContainer}>
          <Text style={errorStyle}>{displayError}</Text>
        </View>
      ) : helperText ? (
        <Text style={helperStyle}>{helperText}</Text>
      ) : null}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    width: '100%',
  },
  label: {
    marginBottom: SPACING.xs,
    letterSpacing: 0.2,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.WHITE,
    borderRadius: RADIUS.md,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.xs,
    shadowColor: COLORS.BLACK,
    shadowOffset: {
      width: 0,
      height: 1,
    },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 1,
  },
  iconContainer: {
    width: 40,
    height: 40,
    borderRadius: RADIUS.sm,
    justifyContent: 'center',
    alignItems: 'center',
    marginHorizontal: SPACING.xs,
  },
  leftIconContainer: {
    marginLeft: 0,
    marginRight: SPACING.sm,
  },
  rightIconContainer: {
    marginLeft: SPACING.sm,
    marginRight: 0,
  },
  input: {
    flex: 1,
    paddingVertical: 0,
    color: COLORS.TEXT_PRIMARY,
    minHeight: 24,
  },
  helperText: {
    marginTop: SPACING.xs,
    marginLeft: SPACING.xs,
  },
  errorContainer: {
    marginTop: SPACING.xs,
    marginLeft: SPACING.xs,
    flexDirection: 'row',
    alignItems: 'center',
  },
  charCount: {
    fontFamily: TYPOGRAPHY.fontFamily.bodyRegular,
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: COLORS.TEXT_MUTED,
    marginLeft: SPACING.sm,
  },
});

/**
 * RadioField Component
 *
 * Component radio buttons cho phép chọn 1 trong nhiều options
 *
 * @example
 * // Sử dụng cơ bản (Vertical layout)
 * const [gender, setGender] = useState<string | number>();
 * const genderOptions = [
 *   { label: 'Nam', value: 'male' },
 *   { label: 'Nữ', value: 'female' },
 *   { label: 'Khác', value: 'other' },
 * ];
 *
 * <RadioField
 *   label="Giới tính"
 *   options={genderOptions}
 *   value={gender}
 *   onChange={setGender}
 * />
 *
 * @example
 * // Horizontal layout với validation
 * <RadioField
 *   label="Phương thức thanh toán"
 *   options={paymentOptions}
 *   value={paymentMethod}
 *   onChange={setPaymentMethod}
 *   horizontal
 *   validators={[
 *     {
 *       validate: (value) => value !== undefined,
 *       message: 'Vui lòng chọn phương thức thanh toán'
 *     }
 *   ]}
 * />
 *
 * @example
 * // Variant Secondary với name attribute
 * <RadioField
 *   variant={RadioVariant.Secondary}
 *   name="subscription-plan"
 *   label="Gói đăng ký"
 *   options={subscriptionOptions}
 *   value={plan}
 *   onChange={setPlan}
 *   helperText="Chọn gói phù hợp với bạn"
 * />
 *
 * @example
 * // Disabled state
 * <RadioField
 *   label="Tuỳ chọn"
 *   options={options}
 *   value={selectedOption}
 *   onChange={setSelectedOption}
 *   disabled
 * />
 */

import { COLORS, RADIUS, SPACING, TYPOGRAPHY } from '@/presentation/theme';
import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ViewStyle,
  TextStyle,
} from 'react-native';

export enum RadioVariant {
  Primary = 'Primary',
  Secondary = 'Secondary',
}

export interface ValidationRule {
  validate: (value: string | number | undefined) => boolean;
  message: string;
}

export interface RadioOption {
  label: string;
  value: string | number;
}

interface RadioFieldProps {
  options: RadioOption[];
  value?: string | number;
  onChange: (value: string | number) => void;
  label?: string;
  disabled?: boolean;
  error?: string;
  helperText?: string;
  variant?: RadioVariant;
  validators?: ValidationRule[];
  style?: ViewStyle;
  horizontal?: boolean;
}

export const RadioField: React.FC<RadioFieldProps> = ({
  options,
  value,
  onChange,
  label,
  disabled = false,
  error,
  helperText,
  variant = RadioVariant.Primary,
  validators = [],
  style,
  horizontal = false,
}) => {
  const [internalError, setInternalError] = useState<string>('');

  const handleSelect = (selectedValue: string | number) => {
    if (disabled) return;

    setInternalError('');

    // Chạy validation rules
    if (validators.length > 0) {
      for (const rule of validators) {
        if (!rule.validate(selectedValue)) {
          setInternalError(rule.message);
          break;
        }
      }
    }

    onChange(selectedValue);
  };

  const getBorderColor = (isSelected: boolean): string => {
    // Disabled state
    if (disabled) {
      return COLORS.GRAY_300;
    }

    // Kiểm tra variant
    if (variant === RadioVariant.Secondary) {
      // Secondary: GRAY_400 mặc định, GRAY_600 khi chọn
      return isSelected ? COLORS.GRAY_600 : COLORS.GRAY_400;
    }

    // Primary: PRIMARY mặc định, SECONDARY khi chọn
    return isSelected ? COLORS.SECONDARY : COLORS.PRIMARY;
  };

  const getCircleBorderColor = (isSelected: boolean): string => {
    // Disabled state
    if (disabled) {
      return COLORS.GRAY_400;
    }

    // Kiểm tra variant
    if (variant === RadioVariant.Secondary) {
      return isSelected ? COLORS.GRAY_600 : COLORS.GRAY_400;
    }

    return isSelected ? COLORS.SECONDARY : COLORS.PRIMARY;
  };

  const getCircleInnerColor = (): string => {
    if (variant === RadioVariant.Secondary) {
      return COLORS.GRAY_600;
    }
    return COLORS.SECONDARY;
  };

  const displayError = error || internalError;

  const containerStyle: ViewStyle = {
    ...styles.container,
    ...style,
  };

  const labelStyle: TextStyle = {
    ...styles.label,
    fontFamily: TYPOGRAPHY.fontFamily.bodyBold,
    fontSize: TYPOGRAPHY.fontSize.sm,
    color: COLORS.TEXT_PRIMARY,
  };

  const optionsContainerStyle: ViewStyle = {
    ...styles.optionsContainer,
    flexDirection: horizontal ? 'row' : 'column',
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

  return (
    <View style={containerStyle}>
      {label && <Text style={labelStyle}>{label}</Text>}

      <View style={optionsContainerStyle}>
        {options.map((option, index) => {
          const isSelected = option.value === value;
          const isDisabled = disabled;

          return (
            <TouchableOpacity
              key={option.value}
              style={[
                styles.radioOption,
                {
                  borderColor: getBorderColor(isSelected),
                },
                horizontal && index > 0 && styles.radioOptionHorizontalSpacing,
                isDisabled && styles.radioOptionDisabled,
              ]}
              onPress={() => handleSelect(option.value)}
              disabled={isDisabled}
              activeOpacity={0.7}
            >
              <View
                style={[
                  styles.radioCircle,
                  {
                    borderColor: getCircleBorderColor(isSelected),
                  },
                  isDisabled && styles.radioCircleDisabled,
                ]}
              >
                {isSelected && (
                  <View
                    style={[
                      styles.radioInner,
                      {
                        backgroundColor: getCircleInnerColor(),
                      },
                    ]}
                  />
                )}
              </View>

              <Text
                style={[
                  styles.radioLabel,
                  isDisabled && styles.radioLabelDisabled,
                ]}
              >
                {option.label}
              </Text>
            </TouchableOpacity>
          );
        })}
      </View>

      {displayError ? (
        <Text style={errorStyle}>{displayError}</Text>
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
    marginBottom: SPACING.sm,
  },
  optionsContainer: {
    gap: SPACING.sm,
  },
  radioOption: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: SPACING.sm,
    backgroundColor: COLORS.WHITE,
    borderWidth: 1.5,
    borderRadius: RADIUS.sm,
  },
  radioOptionHorizontalSpacing: {
    marginLeft: SPACING.sm,
  },
  radioOptionDisabled: {
    opacity: 0.5,
    borderColor: COLORS.GRAY_300,
  },
  radioCircle: {
    width: 20,
    height: 20,
    borderRadius: 10,
    borderWidth: 2,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: SPACING.sm,
  },
  radioCircleDisabled: {
    borderColor: COLORS.GRAY_400,
  },
  radioInner: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  radioLabel: {
    flex: 1,
    fontFamily: TYPOGRAPHY.fontFamily.bodyRegular,
    fontSize: TYPOGRAPHY.fontSize.base,
    color: COLORS.TEXT_PRIMARY,
  },
  radioLabelDisabled: {
    color: COLORS.TEXT_MUTED,
  },
  helperText: {
    marginTop: SPACING.xs,
  },
});

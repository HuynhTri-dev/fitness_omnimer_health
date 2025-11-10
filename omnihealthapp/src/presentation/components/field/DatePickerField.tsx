/**
 * DatePickerField Component
 *
 * Component chọn ngày/giờ với native DateTimePicker
 *
 * @example
 * // Chọn ngày sinh
 * const [birthDate, setBirthDate] = useState<Date>();
 *
 * <DatePickerField
 *   label="Ngày sinh"
 *   value={birthDate}
 *   onChange={setBirthDate}
 *   placeholder="Chọn ngày sinh"
 *   format="DD/MM/YYYY"
 *   maxDate={new Date()}
 * />
 *
 * @example
 * // Chọn giờ với validation
 * <DatePickerField
 *   label="Giờ hẹn"
 *   value={appointmentTime}
 *   onChange={setAppointmentTime}
 *   mode="time"
 *   format="HH:mm"
 *   validators={[
 *     {
 *       validate: (date) => {
 *         if (!date) return false;
 *         const hours = date.getHours();
 *         return hours >= 8 && hours <= 17;
 *       },
 *       message: 'Giờ hẹn phải từ 8h-17h'
 *     }
 *   ]}
 * />
 *
 * @example
 * // Datetime picker với variant Secondary
 * <DatePickerField
 *   variant={DatePickerVariant.Secondary}
 *   label="Ngày và giờ"
 *   value={datetime}
 *   onChange={setDatetime}
 *   mode="datetime"
 *   format="DD/MM/YYYY HH:mm"
 *   minDate={new Date()}
 *   helperText="Chọn thời gian trong tương lai"
 * />
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ViewStyle,
  TextStyle,
  Platform,
} from 'react-native';
import DateTimePicker, {
  DateTimePickerEvent,
} from '@react-native-community/datetimepicker';
import { COLORS, RADIUS, SPACING, TYPOGRAPHY } from '@/presentation/theme';

export enum DatePickerVariant {
  Primary = 'Primary',
  Secondary = 'Secondary',
}

export type DatePickerMode = 'date' | 'time' | 'datetime';

export interface ValidationRule {
  validate: (value: Date | undefined) => boolean;
  message: string;
}

interface DatePickerFieldProps {
  leftIcon?: React.ReactNode;
  value?: Date;
  onChange: (date: Date) => void;
  label?: string;
  placeholder?: string;
  format?: string;
  minDate?: Date;
  maxDate?: Date;
  mode?: DatePickerMode;
  disabled?: boolean;
  error?: string;
  helperText?: string;
  variant?: DatePickerVariant;
  validators?: ValidationRule[];
  style?: ViewStyle;
}

// Calendar Icon - Icon lịch
const CalendarIcon = () => (
  <View style={styles.calendarIcon}>
    <View style={styles.calendarTop} />
    <View style={styles.calendarBody}>
      <View style={styles.calendarDot} />
      <View style={styles.calendarDot} />
      <View style={styles.calendarDot} />
      <View style={styles.calendarDot} />
    </View>
  </View>
);

export const DatePickerField: React.FC<DatePickerFieldProps> = ({
  leftIcon,
  value,
  onChange,
  label,
  placeholder = 'Chọn ngày',
  format = 'DD/MM/YYYY',
  minDate,
  maxDate,
  mode = 'date',
  disabled = false,
  error,
  helperText,
  variant = DatePickerVariant.Primary,
  validators = [],
  style,
}) => {
  const [show, setShow] = useState(false);
  const [internalError, setInternalError] = useState<string>('');

  const handleChange = (event: DateTimePickerEvent, selectedDate?: Date) => {
    // iOS: giữ picker mở, Android: đóng picker
    setShow(Platform.OS === 'ios');

    if (selectedDate) {
      setInternalError('');

      // Chạy validation rules
      if (validators.length > 0) {
        for (const rule of validators) {
          if (!rule.validate(selectedDate)) {
            setInternalError(rule.message);
            break;
          }
        }
      }

      onChange(selectedDate);
    }
  };

  // Format date theo định dạng được chọn
  const formatDate = (date: Date): string => {
    if (!date) return '';

    const day = String(date.getDate()).padStart(2, '0');
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const year = date.getFullYear();
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');

    switch (format) {
      case 'DD/MM/YYYY':
        return `${day}/${month}/${year}`;
      case 'YYYY-MM-DD':
        return `${year}-${month}-${day}`;
      case 'MM/DD/YYYY':
        return `${month}/${day}/${year}`;
      case 'HH:mm':
        return `${hours}:${minutes}`;
      case 'DD/MM/YYYY HH:mm':
        return `${day}/${month}/${year} ${hours}:${minutes}`;
      default:
        return `${day}/${month}/${year}`;
    }
  };

  const getBorderColor = (): string => {
    // Disabled state
    if (disabled) {
      return COLORS.GRAY_300;
    }

    // Error state
    if (error || internalError) {
      return COLORS.ERROR;
    }

    // Kiểm tra variant
    if (variant === DatePickerVariant.Secondary) {
      // Secondary: GRAY_400 mặc định, GRAY_600 khi mở/có value
      if (show || value) {
        return COLORS.GRAY_600;
      }
      return COLORS.GRAY_400;
    }

    // Primary: PRIMARY mặc định, SECONDARY khi mở/có value
    if (show || value) {
      return COLORS.SECONDARY;
    }
    return COLORS.PRIMARY;
  };

  const displayError = error || internalError;

  const containerStyle: ViewStyle = {
    ...styles.container,
    ...style,
  };

  const datePickerStyle: ViewStyle = {
    ...styles.datePicker,
    borderColor: getBorderColor(),
    opacity: disabled ? 0.5 : 1,
  };

  const labelStyle: TextStyle = {
    ...styles.label,
    fontFamily: TYPOGRAPHY.fontFamily.bodyBold,
    fontSize: TYPOGRAPHY.fontSize.sm,
    color: COLORS.TEXT_PRIMARY,
  };

  const dateTextStyle: TextStyle = {
    ...styles.dateText,
    fontFamily: TYPOGRAPHY.fontFamily.bodyRegular,
    fontSize: TYPOGRAPHY.fontSize.base,
    color: value ? COLORS.TEXT_PRIMARY : COLORS.TEXT_MUTED,
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

      <TouchableOpacity
        style={datePickerStyle}
        onPress={() => !disabled && setShow(true)}
        disabled={disabled}
        activeOpacity={0.7}
      >
        {leftIcon && <View style={styles.leftIconContainer}>{leftIcon}</View>}

        <Text style={dateTextStyle} numberOfLines={1}>
          {value ? formatDate(value) : placeholder}
        </Text>

        <View style={styles.rightIconContainer}>
          <CalendarIcon />
        </View>
      </TouchableOpacity>

      {displayError ? (
        <Text style={errorStyle}>{displayError}</Text>
      ) : helperText ? (
        <Text style={helperStyle}>{helperText}</Text>
      ) : null}

      {show && (
        <DateTimePicker
          value={value || new Date()}
          mode={mode}
          display={Platform.OS === 'ios' ? 'spinner' : 'default'}
          onChange={handleChange}
          minimumDate={minDate}
          maximumDate={maxDate}
        />
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    width: '100%',
  },
  label: {
    marginBottom: SPACING.xs,
  },
  datePicker: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.WHITE,
    borderWidth: 1.5,
    borderRadius: RADIUS.sm,
    paddingHorizontal: SPACING.md,
    minHeight: 48,
  },
  leftIconContainer: {
    marginRight: SPACING.sm,
  },
  dateText: {
    flex: 1,
  },
  rightIconContainer: {
    marginLeft: SPACING.sm,
  },
  calendarIcon: {
    width: 18,
    height: 18,
  },
  calendarTop: {
    height: 3,
    backgroundColor: COLORS.TEXT_SECONDARY,
    borderTopLeftRadius: 2,
    borderTopRightRadius: 2,
    marginBottom: 2,
  },
  calendarBody: {
    flex: 1,
    borderWidth: 1.5,
    borderColor: COLORS.TEXT_SECONDARY,
    borderRadius: 2,
    flexDirection: 'row',
    flexWrap: 'wrap',
    padding: 2,
  },
  calendarDot: {
    width: 2,
    height: 2,
    backgroundColor: COLORS.TEXT_SECONDARY,
    borderRadius: 1,
    margin: 1,
  },
  helperText: {
    marginTop: SPACING.xs,
  },
});

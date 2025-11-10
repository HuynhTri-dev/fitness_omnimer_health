/**
 * SingleSelectBox Component
 *
 * Component dropdown select cho phép chọn 1 giá trị từ danh sách
 *
 * @example
 * // Sử dụng cơ bản
 * const [country, setCountry] = useState<string | number>();
 * const options = [
 *   { label: 'Việt Nam', value: 'vn' },
 *   { label: 'Thái Lan', value: 'th' },
 * ];
 *
 * <SingleSelectBox
 *   label="Quốc gia"
 *   value={country}
 *   options={options}
 *   onChange={setCountry}
 *   placeholder="Chọn quốc gia"
 * />
 *
 * @example
 * // Với tìm kiếm và validation
 * <SingleSelectBox
 *   label="Tỉnh/Thành phố"
 *   value={city}
 *   options={cityOptions}
 *   onChange={setCity}
 *   searchable
 *   validators={[
 *     {
 *       validate: (value) => value !== undefined,
 *       message: 'Vui lòng chọn tỉnh/thành phố'
 *     }
 *   ]}
 *   helperText="Chọn nơi bạn đang sinh sống"
 * />
 *
 * @example
 * // Với icon và variant Secondary
 * <SingleSelectBox
 *   variant={SelectVariant.Secondary}
 *   leftIcon={<LocationIcon />}
 *   label="Địa điểm"
 *   value={location}
 *   options={locationOptions}
 *   onChange={setLocation}
 *   maxHeight={400}
 * />
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  Modal,
  FlatList,
  StyleSheet,
  ViewStyle,
  TextStyle,
  TextInput,
} from 'react-native';
import { COLORS, RADIUS, SPACING, TYPOGRAPHY } from '@/presentation/theme';

export enum SelectVariant {
  Primary = 'Primary',
  Secondary = 'Secondary',
}

export interface ValidationRule {
  validate: (value: string | number | undefined) => boolean;
  message: string;
}

export interface SelectOption {
  label: string;
  value: string | number;
}

interface SingleSelectBoxProps {
  leftIcon?: React.ReactNode;
  value?: string | number;
  options: SelectOption[];
  onChange: (value: string | number) => void;
  placeholder?: string;
  label?: string;
  disabled?: boolean;
  error?: string;
  helperText?: string;
  variant?: SelectVariant;
  autoFocus?: boolean;
  validators?: ValidationRule[];
  searchable?: boolean;
  maxHeight?: number;
  style?: ViewStyle;
}

// Chevron Down Icon - Mũi tên chỉ xuống
const ChevronDownIcon = () => (
  <View style={styles.chevronIcon}>
    <View style={styles.chevronLine} />
  </View>
);

export const SingleSelectBox: React.FC<SingleSelectBoxProps> = ({
  leftIcon,
  value,
  options,
  onChange,
  placeholder = 'Chọn một giá trị',
  label,
  disabled = false,
  error,
  helperText,
  variant = SelectVariant.Primary,
  autoFocus = false,
  validators = [],
  searchable = false,
  maxHeight = 300,
  style,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [internalError, setInternalError] = useState<string>('');

  const selectedOption = options.find(opt => opt.value === value);

  // Lọc options khi có searchable
  const filteredOptions = searchable
    ? options.filter(opt =>
        opt.label.toLowerCase().includes(searchQuery.toLowerCase()),
      )
    : options;

  const handleSelect = (selectedValue: string | number) => {
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
    setIsOpen(false);
    setSearchQuery('');
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
    if (variant === SelectVariant.Secondary) {
      // Secondary: GRAY_400 mặc định, GRAY_600 khi mở/có value
      if (isOpen || value !== undefined) {
        return COLORS.GRAY_600;
      }
      return COLORS.GRAY_400;
    }

    // Primary: PRIMARY mặc định, SECONDARY khi mở/có value
    if (isOpen || value !== undefined) {
      return COLORS.SECONDARY;
    }
    return COLORS.PRIMARY;
  };

  const displayError = error || internalError;

  const containerStyle: ViewStyle = {
    ...styles.container,
    ...style,
  };

  const selectBoxStyle: ViewStyle = {
    ...styles.selectBox,
    borderColor: getBorderColor(),
    opacity: disabled ? 0.5 : 1,
  };

  const labelStyle: TextStyle = {
    ...styles.label,
    fontFamily: TYPOGRAPHY.fontFamily.bodyBold,
    fontSize: TYPOGRAPHY.fontSize.sm,
    color: COLORS.TEXT_PRIMARY,
  };

  const selectedTextStyle: TextStyle = {
    ...styles.selectedText,
    fontFamily: TYPOGRAPHY.fontFamily.bodyRegular,
    fontSize: TYPOGRAPHY.fontSize.base,
    color: selectedOption ? COLORS.TEXT_PRIMARY : COLORS.TEXT_MUTED,
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
        style={selectBoxStyle}
        onPress={() => !disabled && setIsOpen(true)}
        disabled={disabled}
        activeOpacity={0.7}
      >
        {leftIcon && <View style={styles.leftIconContainer}>{leftIcon}</View>}

        <Text style={selectedTextStyle} numberOfLines={1}>
          {selectedOption ? selectedOption.label : placeholder}
        </Text>

        <View style={styles.rightIconContainer}>
          <ChevronDownIcon />
        </View>
      </TouchableOpacity>

      {displayError ? (
        <Text style={errorStyle}>{displayError}</Text>
      ) : helperText ? (
        <Text style={helperStyle}>{helperText}</Text>
      ) : null}

      <Modal
        visible={isOpen}
        transparent
        animationType="fade"
        onRequestClose={() => setIsOpen(false)}
      >
        <TouchableOpacity
          style={styles.modalOverlay}
          activeOpacity={1}
          onPress={() => setIsOpen(false)}
        >
          <View style={styles.modalContent}>
            {searchable && (
              <View style={styles.searchContainer}>
                <TextInput
                  style={styles.searchInput}
                  placeholder="Tìm kiếm..."
                  value={searchQuery}
                  onChangeText={setSearchQuery}
                  autoFocus={autoFocus}
                  placeholderTextColor={COLORS.TEXT_MUTED}
                />
              </View>
            )}

            <FlatList
              data={filteredOptions}
              keyExtractor={item => String(item.value)}
              style={{ maxHeight }}
              renderItem={({ item }) => {
                const isSelected = item.value === value;
                return (
                  <TouchableOpacity
                    style={[
                      styles.optionItem,
                      isSelected && styles.optionItemSelected,
                    ]}
                    onPress={() => handleSelect(item.value)}
                    activeOpacity={0.7}
                  >
                    <Text
                      style={[
                        styles.optionText,
                        isSelected && styles.optionTextSelected,
                      ]}
                    >
                      {item.label}
                    </Text>
                  </TouchableOpacity>
                );
              }}
              ListEmptyComponent={
                <Text style={styles.emptyText}>Không tìm thấy kết quả</Text>
              }
            />
          </View>
        </TouchableOpacity>
      </Modal>
    </View>
  );
};
const styles = StyleSheet.create({
  container: {
    width: '100%',
  },
  label: {
    marginBottom: SPACING.xs,
    color: COLORS.TEXT_PRIMARY,
    fontFamily: TYPOGRAPHY.fontFamily.bodyBold,
    fontSize: TYPOGRAPHY.fontSize.sm,
  },
  selectBox: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.WHITE,
    borderWidth: 1.5,
    borderRadius: RADIUS.md,
    paddingHorizontal: SPACING.md,
    minHeight: 48,
    shadowColor: COLORS.BLACK,
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 1,
  },
  leftIconContainer: {
    marginRight: SPACING.sm,
  },
  selectedText: {
    flex: 1,
    fontFamily: TYPOGRAPHY.fontFamily.bodyRegular,
    fontSize: TYPOGRAPHY.fontSize.base,
  },
  rightIconContainer: {
    marginLeft: SPACING.sm,
  },
  chevronIcon: {
    width: 18,
    height: 18,
    justifyContent: 'center',
    alignItems: 'center',
  },
  chevronLine: {
    width: 8,
    height: 8,
    borderBottomWidth: 2,
    borderRightWidth: 2,
    borderColor: COLORS.TEXT_SECONDARY,
    transform: [{ rotate: '45deg' }],
    marginTop: -3,
  },
  helperText: {
    marginTop: SPACING.xs,
    fontFamily: TYPOGRAPHY.fontFamily.bodyRegular,
    fontSize: TYPOGRAPHY.fontSize.xs,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.35)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    width: '90%',
    maxHeight: '80%',
    backgroundColor: COLORS.WHITE,
    borderRadius: RADIUS.lg,
    overflow: 'hidden',
    shadowColor: COLORS.BLACK,
    shadowOpacity: 0.15,
    shadowRadius: 6,
    elevation: 4,
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: SPACING.md,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.BORDER,
  },
  searchInput: {
    flex: 1,
    paddingVertical: SPACING.md,
    fontFamily: TYPOGRAPHY.fontFamily.bodyRegular,
    fontSize: TYPOGRAPHY.fontSize.base,
    color: COLORS.TEXT_PRIMARY,
  },
  optionItem: {
    paddingVertical: SPACING.md,
    paddingHorizontal: SPACING.lg,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.GRAY_200,
    backgroundColor: COLORS.WHITE,
  },
  optionItemSelected: {
    backgroundColor: COLORS.SECONDARY,
  },
  optionText: {
    fontFamily: TYPOGRAPHY.fontFamily.bodyRegular,
    fontSize: TYPOGRAPHY.fontSize.base,
    color: COLORS.TEXT_PRIMARY,
  },
  optionTextSelected: {
    color: COLORS.WHITE,
    fontFamily: TYPOGRAPHY.fontFamily.bodyBold,
  },
  optionItemActive: {
    backgroundColor: COLORS.GRAY_200,
  },
  emptyText: {
    padding: SPACING.lg,
    textAlign: 'center',
    fontFamily: TYPOGRAPHY.fontFamily.bodyRegular,
    fontSize: TYPOGRAPHY.fontSize.sm,
    color: COLORS.TEXT_MUTED,
  },
});

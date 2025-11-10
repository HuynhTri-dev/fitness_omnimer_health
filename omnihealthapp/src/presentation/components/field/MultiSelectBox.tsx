import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  Modal,
  Animated,
  FlatList,
  StyleSheet,
  TextInput,
  ScrollView,
  ViewStyle,
  TextStyle,
} from 'react-native';
import { COLORS, RADIUS, SPACING, TYPOGRAPHY } from '@/presentation/theme';

export enum SelectVariant {
  Primary = 'Primary',
  Secondary = 'Secondary',
}

export interface ValidationRule {
  validate: (value: Array<string | number>) => boolean;
  message: string;
}

export interface SelectOption {
  label: string;
  value: string | number;
}

interface MultiSelectBoxProps {
  leftIcon?: React.ReactNode;
  value?: Array<string | number>;
  options: SelectOption[];
  onChange: (value: Array<string | number>) => void;
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

export const MultiSelectBox: React.FC<MultiSelectBoxProps> = ({
  leftIcon,
  value = [],
  options,
  onChange,
  placeholder = 'Chọn nhiều giá trị',
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

  // animation cho label
  const animatedFocus = useRef(
    new Animated.Value(value.length > 0 ? 1 : 0),
  ).current;
  const isFocused = isOpen || value.length > 0;

  useEffect(() => {
    Animated.timing(animatedFocus, {
      toValue: isFocused ? 1 : 0,
      duration: 150,
      useNativeDriver: false,
    }).start();
  }, [isFocused]);

  const selectedOptions = options.filter(o => value.includes(o.value));
  const filteredOptions = searchable
    ? options.filter(o =>
        o.label.toLowerCase().includes(searchQuery.toLowerCase()),
      )
    : options;

  const handleToggle = (selectedValue: string | number) => {
    const newValue = value.includes(selectedValue)
      ? value.filter(v => v !== selectedValue)
      : [...value, selectedValue];
    setInternalError('');
    if (validators.length > 0) {
      for (const rule of validators) {
        if (!rule.validate(newValue)) {
          setInternalError(rule.message);
          break;
        }
      }
    }
    onChange(newValue);
  };

  const displayError = error || internalError;

  const getBorderColor = (): string => {
    if (disabled) return COLORS.GRAY_300;
    if (displayError) return COLORS.ERROR;

    if (variant === SelectVariant.Secondary)
      return isOpen ? COLORS.GRAY_600 : COLORS.GRAY_400;
    return isOpen ? COLORS.SECONDARY : COLORS.PRIMARY;
  };

  const labelTop = animatedFocus.interpolate({
    inputRange: [0, 1],
    outputRange: [18, -8],
  });
  const labelFontSize = animatedFocus.interpolate({
    inputRange: [0, 1],
    outputRange: [TYPOGRAPHY.fontSize.base, TYPOGRAPHY.fontSize.xs],
  });
  const labelColor = displayError
    ? COLORS.ERROR
    : isOpen
    ? COLORS.SECONDARY
    : COLORS.TEXT_SECONDARY;

  return (
    <View style={[styles.container, style]}>
      <View style={[styles.inputContainer, { borderColor: getBorderColor() }]}>
        {leftIcon && <View style={styles.leftIcon}>{leftIcon}</View>}

        {/* Floating Label */}
        {label && (
          <Animated.Text
            style={[
              styles.label,
              {
                top: labelTop,
                fontSize: labelFontSize,
                color: labelColor,
                backgroundColor: COLORS.WHITE,
              },
            ]}
          >
            {label}
          </Animated.Text>
        )}

        <TouchableOpacity
          style={styles.touchArea}
          onPress={() => !disabled && setIsOpen(true)}
          activeOpacity={0.8}
        >
          <Text
            style={[
              styles.valueText,
              {
                color: selectedOptions.length
                  ? COLORS.TEXT_PRIMARY
                  : COLORS.TEXT_MUTED,
              },
            ]}
            numberOfLines={1}
          >
            {selectedOptions.length === 0
              ? placeholder
              : selectedOptions.length === 1
              ? selectedOptions[0].label
              : `${selectedOptions.length} mục đã chọn`}
          </Text>
        </TouchableOpacity>
      </View>

      {/* Chips */}
      {selectedOptions.length > 0 && (
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          style={styles.chipsContainer}
        >
          {selectedOptions.map(opt => (
            <View key={opt.value} style={styles.chip}>
              <Text style={styles.chipText}>{opt.label}</Text>
              <TouchableOpacity onPress={() => handleToggle(opt.value)}>
                <Text style={styles.removeChip}>×</Text>
              </TouchableOpacity>
            </View>
          ))}
        </ScrollView>
      )}

      {displayError ? (
        <Text style={styles.errorText}>{displayError}</Text>
      ) : helperText ? (
        <Text style={styles.helperText}>{helperText}</Text>
      ) : null}

      {/* Modal chọn nhiều */}
      <Modal visible={isOpen} transparent animationType="fade">
        <TouchableOpacity
          style={styles.overlay}
          activeOpacity={1}
          onPressOut={() => setIsOpen(false)}
        >
          <View style={styles.modalBox}>
            {searchable && (
              <TextInput
                style={styles.searchInput}
                placeholder="Tìm kiếm..."
                value={searchQuery}
                onChangeText={setSearchQuery}
                placeholderTextColor={COLORS.TEXT_MUTED}
                autoFocus={autoFocus}
              />
            )}

            <FlatList
              data={filteredOptions}
              keyExtractor={item => String(item.value)}
              style={{ maxHeight }}
              renderItem={({ item }) => {
                const selected = value.includes(item.value);
                return (
                  <TouchableOpacity
                    onPress={() => handleToggle(item.value)}
                    style={[styles.option, selected && styles.optionSelected]}
                  >
                    <View style={styles.checkbox}>
                      {selected && <View style={styles.checkboxInner} />}
                    </View>
                    <Text
                      style={[
                        styles.optionText,
                        selected && styles.optionTextSelected,
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
  container: { width: '100%', marginBottom: SPACING.sm },
  inputContainer: {
    borderWidth: 1.5,
    borderRadius: RADIUS.sm,
    minHeight: 56,
    justifyContent: 'center',
    paddingHorizontal: SPACING.md,
    backgroundColor: COLORS.WHITE,
  },
  leftIcon: { position: 'absolute', left: SPACING.md },
  label: {
    position: 'absolute',
    left: SPACING.md + 4,
    paddingHorizontal: 4,
    zIndex: 1,
  },
  touchArea: { paddingVertical: 8 },
  valueText: {
    fontFamily: TYPOGRAPHY.fontFamily.bodyRegular,
    fontSize: TYPOGRAPHY.fontSize.base,
  },
  chipsContainer: {
    marginTop: SPACING.xs,
  },
  chip: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.PRIMARY,
    borderRadius: RADIUS.sm,
    paddingHorizontal: SPACING.sm,
    paddingVertical: 4,
    marginRight: SPACING.xs,
  },
  chipText: {
    color: COLORS.WHITE,
    fontSize: TYPOGRAPHY.fontSize.xs,
    marginRight: 4,
  },
  removeChip: {
    color: COLORS.WHITE,
    fontWeight: 'bold',
    fontSize: 14,
  },
  errorText: {
    color: COLORS.ERROR,
    fontSize: TYPOGRAPHY.fontSize.xs,
    marginTop: SPACING.xs,
  },
  helperText: {
    color: COLORS.TEXT_SECONDARY,
    fontSize: TYPOGRAPHY.fontSize.xs,
    marginTop: SPACING.xs,
  },
  overlay: {
    flex: 1,
    backgroundColor: COLORS.OVERLAY,
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalBox: {
    width: '85%',
    backgroundColor: COLORS.WHITE,
    borderRadius: RADIUS.md,
    overflow: 'hidden',
  },
  searchInput: {
    padding: SPACING.md,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.BORDER,
    fontSize: TYPOGRAPHY.fontSize.base,
  },
  option: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: SPACING.md,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.BORDER,
  },
  optionSelected: { backgroundColor: COLORS.PRIMARY },
  optionText: {
    flex: 1,
    color: COLORS.TEXT_PRIMARY,
  },
  optionTextSelected: { fontWeight: 'bold', color: COLORS.PRIMARY },
  checkbox: {
    width: 18,
    height: 18,
    borderWidth: 2,
    borderColor: COLORS.PRIMARY,
    borderRadius: 4,
    marginRight: SPACING.sm,
    alignItems: 'center',
    justifyContent: 'center',
  },
  checkboxInner: {
    width: 10,
    height: 10,
    backgroundColor: COLORS.PRIMARY,
    borderRadius: 2,
  },
  emptyText: {
    textAlign: 'center',
    color: COLORS.TEXT_MUTED,
    padding: SPACING.md,
  },
});

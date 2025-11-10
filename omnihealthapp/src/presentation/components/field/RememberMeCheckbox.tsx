/**
 * RememberMeCheckbox Component
 *
 * Checkbox component với animation
 */

import React from 'react';
import {
  TouchableOpacity,
  View,
  Text,
  StyleSheet,
  Animated,
} from 'react-native';
import { FontAwesomeIcon } from '@fortawesome/react-native-fontawesome';
import { faCheck } from '@fortawesome/free-solid-svg-icons';
import { COLORS, SPACING, TYPOGRAPHY } from '@/presentation/theme';

interface RememberMeCheckboxProps {
  checked: boolean;
  onToggle: () => void;
}

export const RememberMeCheckbox: React.FC<RememberMeCheckboxProps> = ({
  checked,
  onToggle,
}) => {
  const [scaleAnim] = React.useState(new Animated.Value(checked ? 1 : 0));

  React.useEffect(() => {
    Animated.spring(scaleAnim, {
      toValue: checked ? 1 : 0,
      friction: 3,
      useNativeDriver: true,
    }).start();
  }, [checked]);

  return (
    <TouchableOpacity
      style={styles.container}
      onPress={onToggle}
      activeOpacity={0.7}
    >
      <View style={[styles.checkbox, checked && styles.checkboxChecked]}>
        <Animated.View
          style={{
            transform: [{ scale: scaleAnim }],
          }}
        >
          {checked && (
            <FontAwesomeIcon icon={faCheck} size={14} color={COLORS.WHITE} />
          )}
        </Animated.View>
      </View>
      <Text style={styles.label}>Ghi nhớ đăng nhập</Text>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  checkbox: {
    width: 22,
    height: 22,
    borderWidth: 2,
    borderColor: '#4A90E2',
    borderRadius: 6,
    marginRight: SPACING.xs,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: COLORS.WHITE,
  },
  checkboxChecked: {
    backgroundColor: '#4A90E2',
    borderColor: '#4A90E2',
  },
  label: {
    fontFamily: TYPOGRAPHY.fontFamily.bodyRegular,
    fontSize: TYPOGRAPHY.fontSize.sm,
    color: COLORS.TEXT_PRIMARY,
  },
});

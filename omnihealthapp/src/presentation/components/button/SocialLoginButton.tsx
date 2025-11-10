/**
 * SocialLoginButton Component
 *
 * Button cho đăng nhập bằng mạng xã hội
 */

import React from 'react';
import { TouchableOpacity, Text, StyleSheet, View } from 'react-native';
import { COLORS, TYPOGRAPHY } from '@/presentation/theme';

type SocialType = 'google' | 'facebook' | 'apple';

interface SocialLoginButtonProps {
  type: SocialType;
  onPress: () => void;
}

const socialConfig = {
  google: {
    icon: 'G',
    color: '#DB4437',
    label: 'Google',
  },
  facebook: {
    icon: 'f',
    color: '#4267B2',
    label: 'Facebook',
  },
  apple: {
    icon: '',
    color: '#000000',
    label: 'Apple',
  },
};

export const SocialLoginButton: React.FC<SocialLoginButtonProps> = ({
  type,
  onPress,
}) => {
  const config = socialConfig[type];

  return (
    <TouchableOpacity
      style={styles.button}
      onPress={onPress}
      activeOpacity={0.7}
    >
      <View style={styles.iconContainer}>
        <Text style={[styles.icon, { color: config.color }]}>
          {config.icon}
        </Text>
      </View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  button: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: COLORS.WHITE,
    borderWidth: 1.5,
    borderColor: COLORS.BORDER,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  iconContainer: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  icon: {
    fontFamily: TYPOGRAPHY.fontFamily.bodyBold,
    fontSize: 24,
    fontWeight: 'bold',
  },
});

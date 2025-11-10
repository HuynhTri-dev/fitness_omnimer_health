/**
 * useLoginForm Hook
 *
 * Custom hook quản lý logic của form đăng nhập
 */

import { useState } from 'react';
import { Alert } from 'react-native';
import {
  validateField,
  emailValidators,
  passwordValidators,
} from '@utils/validator/auth.validator';

export const useLoginForm = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const [loading, setLoading] = useState(false);

  const validateForm = (): boolean => {
    // Validate email
    const emailError = validateField(email, emailValidators);
    if (emailError) {
      Alert.alert('Lỗi', emailError);
      return false;
    }

    // Validate password
    const passwordError = validateField(password, passwordValidators);
    if (passwordError) {
      Alert.alert('Lỗi', passwordError);
      return false;
    }

    return true;
  };

  const handleLogin = async () => {
    // Validate form
    if (!validateForm()) {
      return;
    }

    setLoading(true);

    try {
      console.log('Login successful:', {
        email,
        password,
        rememberMe,
      });

      // TODO: Navigate to home screen
      Alert.alert('Thành công', 'Đăng nhập thành công!');
    } catch (error) {
      console.error('Login error:', error);
      Alert.alert('Lỗi', 'Đăng nhập thất bại. Vui lòng thử lại.');
    } finally {
      setLoading(false);
    }
  };

  return {
    email,
    setEmail,
    password,
    setPassword,
    showPassword,
    setShowPassword,
    rememberMe,
    setRememberMe,
    loading,
    handleLogin,
  };
};

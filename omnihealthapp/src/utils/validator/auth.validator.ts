/**
 * Validation Utilities
 *
 * Các hàm và rules để validate form inputs
 */

export interface ValidationRule {
  validate: (value: string) => boolean;
  message: string;
}

/**
 * Validate một field dựa trên các rules
 */
export const validateField = (
  value: string,
  rules: ValidationRule[],
): string => {
  for (const rule of rules) {
    if (!rule.validate(value)) {
      return rule.message;
    }
  }
  return '';
};

/**
 * Email validation rules
 */
export const emailValidators: ValidationRule[] = [
  {
    validate: (value: string) => value.length > 0,
    message: 'Email không được để trống',
  },
  {
    validate: (value: string) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value),
    message: 'Email không hợp lệ',
  },
];

/**
 * Password validation rules
 */
export const passwordValidators: ValidationRule[] = [
  {
    validate: (value: string) => value.length > 0,
    message: 'Mật khẩu không được để trống',
  },
  {
    validate: (value: string) => value.length >= 6,
    message: 'Mật khẩu phải có ít nhất 6 ký tự',
  },
];

/**
 * Phone number validation rules
 */
export const phoneValidators: ValidationRule[] = [
  {
    validate: (value: string) => value.length > 0,
    message: 'Số điện thoại không được để trống',
  },
  {
    validate: (value: string) => /^[0-9]{10,11}$/.test(value),
    message: 'Số điện thoại không hợp lệ',
  },
];

/**
 * Required field validation
 */
export const requiredValidator: ValidationRule = {
  validate: (value: string) => value.trim().length > 0,
  message: 'Trường này không được để trống',
};

/**
 * Min length validation
 */
export const minLengthValidator = (minLength: number): ValidationRule => ({
  validate: (value: string) => value.length >= minLength,
  message: `Phải có ít nhất ${minLength} ký tự`,
});

/**
 * Max length validation
 */
export const maxLengthValidator = (maxLength: number): ValidationRule => ({
  validate: (value: string) => value.length <= maxLength,
  message: `Không được vượt quá ${maxLength} ký tự`,
});

/**
 * URL validation
 */
export const urlValidator: ValidationRule = {
  validate: (value: string) => {
    try {
      new URL(value);
      return true;
    } catch {
      return false;
    }
  },
  message: 'URL không hợp lệ',
};

/**
 * Number validation
 */
export const numberValidator: ValidationRule = {
  validate: (value: string) => !isNaN(Number(value)) && value.trim() !== '',
  message: 'Giá trị phải là số',
};

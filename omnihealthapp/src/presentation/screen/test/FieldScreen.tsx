// FormExample.tsx - Ví dụ sử dụng tất cả form fields
import React, { useState } from 'react';
import { View, ScrollView, StyleSheet } from 'react-native';
import {
  TextField,
  ValidationRule as TextValidationRule,
} from '@presentation/components/field/TextField';
import {
  SingleSelectBox,
  SelectOption,
  ValidationRule as SelectValidationRule,
} from '@presentation/components/field/SingleSelectBox';
import {
  MultiSelectBox,
  ValidationRule as MultiSelectValidationRule,
} from '@presentation/components/field/MultiSelectBox';
import {
  DatePickerField,
  ValidationRule as DateValidationRule,
} from '@presentation/components/field/DatePickerField';
import {
  RadioField,
  RadioOption,
  ValidationRule as RadioValidationRule,
} from '@presentation/components/field/RadioField';
import {
  ButtonPrimary,
  ButtonVariant,
} from '@presentation/components/button/ButtonPrimary';
import { COLORS, SPACING } from '@/presentation/theme';

// Icon example - replace with actual icons
const UserIcon = () => (
  <View style={{ width: 20, height: 20, backgroundColor: COLORS.PRIMARY }} />
);
const EmailIcon = () => (
  <View style={{ width: 20, height: 20, backgroundColor: COLORS.PRIMARY }} />
);

const FormExample = () => {
  // Form state
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [bio, setBio] = useState('');
  const [country, setCountry] = useState<string | number>();
  const [skills, setSkills] = useState<Array<string | number>>([]);
  const [birthDate, setBirthDate] = useState<Date>();
  const [gender, setGender] = useState<string | number>();

  // Options
  const countryOptions: SelectOption[] = [
    { label: 'Việt Nam', value: 'vn' },
    { label: 'Thái Lan', value: 'th' },
    { label: 'Singapore', value: 'sg' },
    { label: 'Malaysia', value: 'my' },
    { label: 'Indonesia', value: 'id' },
  ];

  const skillOptions: SelectOption[] = [
    { label: 'React Native', value: 'react-native' },
    { label: 'TypeScript', value: 'typescript' },
    { label: 'JavaScript', value: 'javascript' },
    { label: 'Node.js', value: 'nodejs' },
    { label: 'Python', value: 'python' },
    { label: 'Java', value: 'java' },
  ];

  const genderOptions: RadioOption[] = [
    { label: 'Nam', value: 'male' },
    { label: 'Nữ', value: 'female' },
    { label: 'Khác', value: 'other' },
  ];

  // Validators
  const nameValidators: TextValidationRule[] = [
    {
      validate: (value: string) => value.length >= 3,
      message: 'Tên phải có ít nhất 3 ký tự',
    },
  ];

  const emailValidators: TextValidationRule[] = [
    {
      validate: (value: string) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value),
      message: 'Email không hợp lệ',
    },
  ];

  const passwordValidators: TextValidationRule[] = [
    {
      validate: (value: string) => value.length >= 8,
      message: 'Mật khẩu phải có ít nhất 8 ký tự',
    },
  ];

  const countryValidators: SelectValidationRule[] = [
    {
      validate: value => value !== undefined,
      message: 'Vui lòng chọn quốc gia',
    },
  ];

  const skillsValidators: MultiSelectValidationRule[] = [
    {
      validate: value => value !== undefined,
      message: 'Vui lòng chọn ít nhất 1 kỹ năng',
    },
  ];

  const birthDateValidators: DateValidationRule[] = [
    {
      validate: value => {
        if (!value) return false;
        const age = new Date().getFullYear() - value.getFullYear();
        return age >= 18;
      },
      message: 'Bạn phải từ 18 tuổi trở lên',
    },
  ];

  const genderValidators: RadioValidationRule[] = [
    {
      validate: value => value !== undefined,
      message: 'Vui lòng chọn giới tính',
    },
  ];

  const handleSubmit = () => {
    console.log({
      name,
      email,
      password,
      bio,
      country,
      skills,
      birthDate,
      gender,
    });
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.form}>
        {/* TextField with icon */}
        <TextField
          label="Họ và tên"
          value={name}
          onChangeText={setName}
          placeholder="Nhập họ và tên"
          leftIcon={<UserIcon />}
          validators={nameValidators}
          helperText="Tên đầy đủ của bạn"
          style={styles.field}
        />

        {/* TextField - Email */}
        <TextField
          label="Email"
          value={email}
          onChangeText={setEmail}
          placeholder="example@email.com"
          keyboardType="email-address"
          leftIcon={<EmailIcon />}
          validators={emailValidators}
          style={styles.field}
        />

        {/* TextField - Password */}
        <TextField
          label="Mật khẩu"
          value={password}
          onChangeText={setPassword}
          placeholder="Nhập mật khẩu"
          secureTextEntry
          validators={passwordValidators}
          helperText="Tối thiểu 8 ký tự"
          style={styles.field}
        />

        {/* TextField - Multiline */}
        <TextField
          label="Giới thiệu"
          value={bio}
          onChangeText={setBio}
          placeholder="Viết vài dòng về bạn..."
          multiline
          numberOfLines={4}
          maxLength={200}
          helperText={`${bio.length}/200 ký tự`}
          style={styles.field}
        />

        {/* SingleSelectBox */}
        <SingleSelectBox
          label="Quốc gia"
          value={country}
          options={countryOptions}
          onChange={setCountry}
          placeholder="Chọn quốc gia"
          searchable
          validators={countryValidators}
          helperText="Chọn quốc gia bạn đang sinh sống"
          style={styles.field}
        />

        {/* MultiSelectBox */}
        <MultiSelectBox
          label="Kỹ năng"
          value={skills}
          options={skillOptions}
          onChange={setSkills}
          placeholder="Chọn kỹ năng của bạn"
          searchable
          validators={skillsValidators}
          helperText="Bạn có thể chọn nhiều kỹ năng"
          style={styles.field}
        />

        {/* DatePickerField */}
        <DatePickerField
          label="Ngày sinh"
          value={birthDate}
          onChange={setBirthDate}
          placeholder="Chọn ngày sinh"
          format="DD/MM/YYYY"
          maxDate={new Date()}
          validators={birthDateValidators}
          helperText="Bạn phải từ 18 tuổi trở lên"
          style={styles.field}
        />

        {/* RadioField - Vertical */}
        <RadioField
          label="Giới tính"
          options={genderOptions}
          value={gender}
          onChange={setGender}
          validators={genderValidators}
          style={styles.field}
        />

        {/* RadioField - Horizontal */}
        <RadioField
          label="Giới tính (Ngang)"
          options={genderOptions}
          value={gender}
          onChange={setGender}
          horizontal
          style={styles.field}
        />

        {/* Submit Button */}
        <ButtonPrimary
          variant={ButtonVariant.PrimarySolid}
          title="Đăng ký"
          onPress={handleSubmit}
          style={styles.submitButton}
        />

        {/* Disabled Example */}
        <TextField
          label="Trường bị khóa"
          value="Không thể chỉnh sửa"
          editable={false}
          helperText="Trường này không thể chỉnh sửa"
          style={styles.field}
        />
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.WHITE,
  },
  form: {
    padding: SPACING.lg,
  },
  field: {
    marginBottom: SPACING.lg,
  },
  submitButton: {
    width: '100%',
    marginTop: SPACING.lg,
  },
});

export default FormExample;

import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:omnihealthmobileflutter/core/constants/enum_constant.dart';
import 'package:omnihealthmobileflutter/core/theme/app_colors.dart';
import 'package:omnihealthmobileflutter/core/theme/app_radius.dart';
import 'package:omnihealthmobileflutter/core/theme/app_spacing.dart';
import 'package:omnihealthmobileflutter/presentation/common/input_fields/custom_text_field.dart';

class InfoAccountForm extends StatelessWidget {
  final TextEditingController fullnameController;
  final TextEditingController birthdayController;
  final GenderEnum? selectedGender;
  final ValueChanged<GenderEnum?> onGenderChanged;
  final VoidCallback onSelectDate;
  final bool isLoading;

  const InfoAccountForm({
    super.key,
    required this.fullnameController,
    required this.birthdayController,
    required this.selectedGender,
    required this.onGenderChanged,
    required this.onSelectDate,
    this.isLoading = false,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        CustomTextField(
          controller: fullnameController,
          label: "Fullname",
          placeholder: "Enter fullname",
          enabled: !isLoading,
          leftIcon: const Icon(Icons.person_outline, color: AppColors.primary),
        ),
        SizedBox(height: AppSpacing.md.h),
        GestureDetector(
          onTap: isLoading ? null : onSelectDate,
          child: AbsorbPointer(
            child: CustomTextField(
              controller: birthdayController,
              label: "Birthday",
              placeholder: "YYYY-MM-DD",
              enabled: !isLoading,
              leftIcon: const Icon(
                Icons.cake_outlined,
                color: AppColors.primary,
              ),
              rightIcon: const Icon(
                Icons.calendar_today,
                color: AppColors.textSecondary,
              ),
            ),
          ),
        ),
        SizedBox(height: AppSpacing.md.h),
        _buildGenderDropdown(),
      ],
    );
  }

  Widget _buildGenderDropdown() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          "Gender",
          style: TextStyle(
            fontSize: 14.sp,
            fontWeight: FontWeight.w700,
            color: AppColors.textPrimary,
          ),
        ),
        SizedBox(height: AppSpacing.xs.h),
        Container(
          padding: EdgeInsets.symmetric(horizontal: 12.w),
          decoration: BoxDecoration(
            color: AppColors.surface,
            borderRadius: BorderRadius.circular(AppRadius.md.r),
            border: Border.all(color: AppColors.gray200, width: 1.5),
          ),
          child: DropdownButtonHideUnderline(
            child: DropdownButton<GenderEnum>(
              value: selectedGender,
              isExpanded: true,
              hint: const Text("Choose gender"),
              icon: const Icon(Icons.arrow_drop_down, color: AppColors.primary),
              items: GenderEnum.values.map((gender) {
                return DropdownMenuItem(
                  value: gender,
                  child: Row(
                    children: [
                      Icon(
                        gender == GenderEnum.male
                            ? Icons.male
                            : gender == GenderEnum.female
                            ? Icons.female
                            : Icons.transgender,
                        size: 20.w,
                        color: AppColors.textSecondary,
                      ),
                      SizedBox(width: 8.w),
                      Text(gender.displayName),
                    ],
                  ),
                );
              }).toList(),
              onChanged: isLoading ? null : onGenderChanged,
            ),
          ),
        ),
      ],
    );
  }
}

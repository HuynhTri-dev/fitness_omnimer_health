import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:omnihealthmobileflutter/core/theme/app_colors.dart';
import 'package:omnihealthmobileflutter/core/theme/app_radius.dart';
import 'package:omnihealthmobileflutter/core/theme/app_spacing.dart';
import 'package:omnihealthmobileflutter/core/theme/app_typography.dart';

class InfoAccountReadOnly extends StatelessWidget {
  final String? email;
  final List<String>? roles;

  const InfoAccountReadOnly({super.key, this.email, this.roles});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.all(AppSpacing.md.w),
      decoration: BoxDecoration(
        color: AppColors.gray100,
        borderRadius: BorderRadius.circular(AppRadius.md.r),
        border: Border.all(color: AppColors.gray300),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildItem(
            icon: Icons.email_outlined,
            label: "Email",
            value: email ?? "Chưa cập nhật",
          ),
          if (roles != null && roles!.isNotEmpty) ...[
            SizedBox(height: AppSpacing.md.h),
            Divider(color: AppColors.gray300, height: 1.h),
            SizedBox(height: AppSpacing.md.h),
            _buildItem(
              icon: Icons.security_outlined,
              label: "Roles",
              value: roles!.join(", "),
              isChip: true,
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildItem({
    required IconData icon,
    required String label,
    required String value,
    bool isChip = false,
  }) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, size: 20.w, color: AppColors.textSecondary),
        SizedBox(width: AppSpacing.sm.w),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                label,
                style: AppTypography.caption.copyWith(
                  color: AppColors.textSecondary,
                ),
              ),
              SizedBox(height: 4.h),
              isChip
                  ? Wrap(
                      spacing: 8.w,
                      runSpacing: 8.h,
                      children: value.split(", ").map((role) {
                        return Container(
                          padding: EdgeInsets.symmetric(
                            horizontal: 8.w,
                            vertical: 4.h,
                          ),
                          decoration: BoxDecoration(
                            color: AppColors.primary.withOpacity(0.1),
                            borderRadius: BorderRadius.circular(AppRadius.sm.r),
                            border: Border.all(
                              color: AppColors.primary.withOpacity(0.3),
                            ),
                          ),
                          child: Text(
                            role,
                            style: AppTypography.bodySmall.copyWith(
                              color: AppColors.primary,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        );
                      }).toList(),
                    )
                  : Text(
                      value,
                      style: AppTypography.bodyMedium.copyWith(
                        color: AppColors.textPrimary,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
            ],
          ),
        ),
      ],
    );
  }
}

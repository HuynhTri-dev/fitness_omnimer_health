import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:omnihealthmobileflutter/core/theme/app_colors.dart';
import 'package:omnihealthmobileflutter/core/theme/app_radius.dart';
import 'package:omnihealthmobileflutter/core/theme/app_spacing.dart';
import 'package:omnihealthmobileflutter/core/theme/app_typography.dart';

/// Widget cho mỗi menu item trong More screen
class MoreMenuItem extends StatelessWidget {
  final IconData icon;
  final String title;
  final String? subtitle;
  final VoidCallback onTap;
  final Color? iconColor;
  final Color? backgroundColor;
  final Widget? trailing;
  final bool showArrow;

  const MoreMenuItem({
    Key? key,
    required this.icon,
    required this.title,
    this.subtitle,
    required this.onTap,
    this.iconColor,
    this.backgroundColor,
    this.trailing,
    this.showArrow = true,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: AppRadius.radiusMd,
      child: Container(
        padding: EdgeInsets.symmetric(
          horizontal: AppSpacing.md,
          vertical: AppSpacing.sm,
        ),
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: AppRadius.radiusMd,
          border: Border.all(
            color: AppColors.border.withOpacity(0.5),
            width: 1,
          ),
        ),
        child: Row(
          children: [
            // Icon container
            Container(
              width: 40.w,
              height: 40.h,
              decoration: BoxDecoration(
                color: backgroundColor ?? AppColors.primary.withOpacity(0.1),
                borderRadius: AppRadius.radiusSm,
              ),
              child: Icon(
                icon,
                color: iconColor ?? AppColors.primary,
                size: 20.sp,
              ),
            ),

            SizedBox(width: AppSpacing.md),

            // Title and subtitle
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: AppTypography.bodyBoldStyle(
                      fontSize: AppTypography.fontSizeBase.sp,
                      color: AppColors.textPrimary,
                    ),
                  ),
                  if (subtitle != null) ...[
                    SizedBox(height: 2.h),
                    Text(
                      subtitle!,
                      style: AppTypography.bodyRegularStyle(
                        fontSize: AppTypography.fontSizeXs.sp,
                        color: AppColors.textSecondary,
                      ),
                    ),
                  ],
                ],
              ),
            ),

            // Trailing widget or arrow
            if (trailing != null)
              trailing!
            else if (showArrow)
              Icon(
                Icons.chevron_right,
                color: AppColors.textSecondary,
                size: 20.sp,
              ),
          ],
        ),
      ),
    );
  }
}

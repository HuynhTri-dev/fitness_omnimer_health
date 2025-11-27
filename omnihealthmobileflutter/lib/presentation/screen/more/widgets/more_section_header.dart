import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:omnihealthmobileflutter/core/theme/app_colors.dart';
import 'package:omnihealthmobileflutter/core/theme/app_spacing.dart';
import 'package:omnihealthmobileflutter/core/theme/app_typography.dart';

/// Widget cho section header trong More screen
class MoreSectionHeader extends StatelessWidget {
  final String title;

  const MoreSectionHeader({Key? key, required this.title}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.only(
        left: AppSpacing.xs,
        bottom: AppSpacing.sm,
        top: AppSpacing.lg,
      ),
      child: Text(
        title,
        style: AppTypography.bodyBoldStyle(
          fontSize: AppTypography.fontSizeSm.sp,
          color: AppColors.textSecondary,
        ),
      ),
    );
  }
}

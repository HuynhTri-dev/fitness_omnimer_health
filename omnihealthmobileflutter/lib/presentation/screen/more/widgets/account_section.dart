import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:omnihealthmobileflutter/core/theme/app_colors.dart';
import 'package:omnihealthmobileflutter/core/theme/app_radius.dart';
import 'package:omnihealthmobileflutter/core/theme/app_spacing.dart';
import 'package:omnihealthmobileflutter/core/theme/app_typography.dart';
import 'package:omnihealthmobileflutter/presentation/screen/more/widgets/more_menu_item.dart';

/// Account section widget for More screen
class AccountSection extends StatelessWidget {
  final VoidCallback onProfileTap;
  final VoidCallback onPremiumTap;

  const AccountSection({
    Key? key,
    required this.onProfileTap,
    required this.onPremiumTap,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        MoreMenuItem(
          icon: Icons.person_outline,
          title: 'Profile',
          subtitle: 'Manage personal info & body metrics',
          onTap: onProfileTap,
        ),
        SizedBox(height: AppSpacing.sm),
        MoreMenuItem(
          icon: Icons.workspace_premium_outlined,
          title: 'Upgrade to Premium',
          subtitle: 'Unlock advanced features',
          iconColor: AppColors.warning,
          backgroundColor: AppColors.warning.withOpacity(0.1),
          onTap: onPremiumTap,
          trailing: Container(
            padding: EdgeInsets.symmetric(
              horizontal: AppSpacing.sm,
              vertical: AppSpacing.xs,
            ),
            decoration: BoxDecoration(
              color: AppColors.warning,
              borderRadius: AppRadius.radiusSm,
            ),
            child: Text(
              'PRO',
              style: AppTypography.bodyBoldStyle(
                fontSize: AppTypography.fontSizeXs.sp,
                color: AppColors.white,
              ),
            ),
          ),
        ),
      ],
    );
  }
}

import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:omnihealthmobileflutter/core/theme/app_colors.dart';
import 'package:omnihealthmobileflutter/core/theme/app_radius.dart';
import 'package:omnihealthmobileflutter/core/theme/app_spacing.dart';
import 'package:omnihealthmobileflutter/core/theme/app_typography.dart';
import 'package:omnihealthmobileflutter/presentation/screen/more/widgets/more_menu_item.dart';
import 'package:omnihealthmobileflutter/presentation/screen/more/profile/verify_account_screen.dart';
import 'package:omnihealthmobileflutter/presentation/screen/auth/info_account/info_account_screen.dart';
import 'package:omnihealthmobileflutter/presentation/screen/more/profile/change_password_screen.dart';

/// Account section widget for More screen
class AccountSection extends StatefulWidget {
  final VoidCallback onPremiumTap;

  const AccountSection({Key? key, required this.onPremiumTap})
    : super(key: key);

  @override
  State<AccountSection> createState() => _AccountSectionState();
}

class _AccountSectionState extends State<AccountSection>
    with SingleTickerProviderStateMixin {
  bool _isExpanded = false;
  late AnimationController _animationController;
  late Animation<double> _expandAnimation;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );
    _expandAnimation = CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeInOut,
    );
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  void _toggleExpanded() {
    setState(() {
      _isExpanded = !_isExpanded;
      if (_isExpanded) {
        _animationController.forward();
      } else {
        _animationController.reverse();
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // Profile Menu Item with Dropdown
        MoreMenuItem(
          icon: Icons.person_outline,
          title: 'Profile',
          subtitle: 'Manage personal info',
          onTap: _toggleExpanded,
          trailing: RotationTransition(
            turns: Tween(begin: 0.0, end: 0.5).animate(_expandAnimation),
            child: Icon(
              Icons.keyboard_arrow_down,
              color: AppColors.textMuted,
              size: 24.sp,
            ),
          ),
        ),

        // Dropdown Menu
        SizeTransition(
          sizeFactor: _expandAnimation,
          child: Container(
            margin: EdgeInsets.only(
              left: AppSpacing.lg,
              top: AppSpacing.xs,
              bottom: AppSpacing.xs,
            ),
            decoration: BoxDecoration(
              border: Border(
                left: BorderSide(
                  color: AppColors.primary.withOpacity(0.3),
                  width: 2,
                ),
              ),
            ),
            child: Column(
              children: [
                _buildDropdownItem(
                  icon: Icons.verified_user_outlined,
                  title: 'Verify Account',
                  subtitle: 'Email & authentication methods',
                  onTap: () =>
                      _navigateToScreen(context, const VerifyAccountScreen()),
                ),
                _buildDropdownItem(
                  icon: Icons.info_outline,
                  title: 'Info Account',
                  subtitle: 'Update basic information',
                  onTap: () =>
                      _navigateToScreen(context, const InfoAccountScreen()),
                ),
                _buildDropdownItem(
                  icon: Icons.lock_outline,
                  title: 'Change Password',
                  subtitle: 'Update your password',
                  onTap: () =>
                      _navigateToScreen(context, const ChangePasswordScreen()),
                ),
              ],
            ),
          ),
        ),

        SizedBox(height: AppSpacing.sm),

        // Premium Menu Item
        MoreMenuItem(
          icon: Icons.workspace_premium_outlined,
          title: 'Upgrade to Premium',
          subtitle: 'Unlock advanced features',
          iconColor: AppColors.warning,
          backgroundColor: AppColors.warning.withOpacity(0.1),
          onTap: widget.onPremiumTap,
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

  Widget _buildDropdownItem({
    required IconData icon,
    required String title,
    required String subtitle,
    required VoidCallback onTap,
  }) {
    return InkWell(
      onTap: onTap,
      child: Container(
        padding: EdgeInsets.symmetric(
          horizontal: AppSpacing.md,
          vertical: AppSpacing.sm,
        ),
        child: Row(
          children: [
            Container(
              padding: EdgeInsets.all(AppSpacing.xs),
              decoration: BoxDecoration(
                color: AppColors.primary.withOpacity(0.1),
                borderRadius: AppRadius.radiusSm,
              ),
              child: Icon(icon, color: AppColors.primary, size: 20.sp),
            ),
            SizedBox(width: AppSpacing.sm),
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
                  SizedBox(height: 2.h),
                  Text(
                    subtitle,
                    style: AppTypography.bodyRegularStyle(
                      fontSize: AppTypography.fontSizeSm.sp,
                      color: AppColors.textSecondary,
                    ),
                  ),
                ],
              ),
            ),
            Icon(
              Icons.arrow_forward_ios,
              color: AppColors.textMuted,
              size: 14.sp,
            ),
          ],
        ),
      ),
    );
  }

  void _navigateToScreen(BuildContext context, Widget screen) {
    Navigator.push(context, MaterialPageRoute(builder: (context) => screen));
  }
}

import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:omnihealthmobileflutter/core/theme/app_colors.dart';
import 'package:omnihealthmobileflutter/core/theme/app_spacing.dart';
import 'package:omnihealthmobileflutter/core/theme/app_typography.dart';
import 'package:omnihealthmobileflutter/core/theme/app_radius.dart';
import 'package:omnihealthmobileflutter/presentation/common/auth/user_header_widget.dart';
import 'package:omnihealthmobileflutter/presentation/common/blocs/auth/authentication_bloc.dart';
import 'package:omnihealthmobileflutter/presentation/common/blocs/auth/authentication_event.dart';
import 'package:omnihealthmobileflutter/presentation/screen/more/bloc/more_bloc.dart';
import 'package:omnihealthmobileflutter/presentation/screen/more/bloc/more_event.dart';
import 'package:omnihealthmobileflutter/presentation/screen/more/bloc/more_state.dart';
import 'package:omnihealthmobileflutter/presentation/screen/more/widgets/more_menu_item.dart';
import 'package:omnihealthmobileflutter/presentation/screen/more/widgets/more_section_header.dart';
import 'package:omnihealthmobileflutter/presentation/screen/more/widgets/account_section.dart';
import 'package:omnihealthmobileflutter/presentation/screen/more/widgets/health_data_section.dart';
import 'package:omnihealthmobileflutter/presentation/screen/more/widgets/settings_section.dart';
import 'package:omnihealthmobileflutter/presentation/screen/more/widgets/support_section.dart';

/// More Screen - Settings & Utilities
/// Includes: Profile, Privacy, Device Connectivity, Theme, Premium, Support
class MoreScreen extends StatelessWidget {
  const MoreScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (context) => MoreBloc()..add(const LoadMoreSettings()),
      child: Scaffold(
        backgroundColor: AppColors.background,
        body: Column(
          children: [
            // User Header
            const UserHeaderWidget(),

            // Content
            Expanded(
              child: BlocBuilder<MoreBloc, MoreState>(
                builder: (context, state) {
                  if (state is MoreLoading) {
                    return const Center(child: CircularProgressIndicator());
                  }

                  if (state is MoreError) {
                    return Center(
                      child: Text(
                        'Error: ${state.message}',
                        style: AppTypography.bodyRegularStyle(
                          color: AppColors.error,
                        ),
                      ),
                    );
                  }

                  if (state is MoreLoaded) {
                    return _buildContent(context, state);
                  }

                  return const SizedBox.shrink();
                },
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildContent(BuildContext context, MoreLoaded state) {
    return SingleChildScrollView(
      padding: EdgeInsets.symmetric(
        horizontal: AppSpacing.md,
        vertical: AppSpacing.sm,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Section: Account
          const MoreSectionHeader(title: 'ACCOUNT'),
          AccountSection(onPremiumTap: () => _handlePremiumTap(context)),

          // Section: Health & Data
          const MoreSectionHeader(title: 'HEALTH & DATA'),
          HealthDataSection(
            onDeviceConnectivityTap: () =>
                _handleDeviceConnectivityTap(context),
            onPrivacyTap: () => _handlePrivacyTap(context),
            onExportDataTap: () => _handleExportDataTap(context),
          ),

          // Section: Settings
          const MoreSectionHeader(title: 'SETTINGS'),
          SettingsSection(
            state: state,
            onThemeTap: () => _handleThemeTap(context),
            onLanguageTap: () => _handleLanguageTap(context),
            onNotificationsTap: () => _handleNotificationsTap(context, state),
          ),

          // Section: Support
          const MoreSectionHeader(title: 'SUPPORT'),
          SupportSection(
            onRateUsTap: () => _handleRateUsTap(context),
            onFeedbackTap: () => _handleFeedbackTap(context),
            onAboutTap: () => _handleAboutTap(context),
          ),

          SizedBox(height: AppSpacing.lg),

          // Logout Button
          _buildLogoutButton(context),

          SizedBox(height: AppSpacing.md),

          // Version Info
          _buildVersionInfo(),

          SizedBox(height: AppSpacing.xl),
        ],
      ),
    );
  }

  Widget _buildLogoutButton(BuildContext context) {
    return MoreMenuItem(
      icon: Icons.logout_outlined,
      title: 'Logout',
      subtitle: 'Sign out from current account',
      iconColor: AppColors.danger,
      backgroundColor: AppColors.danger.withOpacity(0.1),
      showArrow: false,
      onTap: () => _handleLogoutTap(context),
    );
  }

  Widget _buildVersionInfo() {
    return Center(
      child: Text(
        'OmniMer Health v1.0.0',
        style: AppTypography.bodyRegularStyle(
          fontSize: AppTypography.fontSizeXs.sp,
          color: AppColors.textMuted,
        ),
      ),
    );
  }

  // Navigation handlers
  void _handlePremiumTap(BuildContext context) {
    _showComingSoonSnackbar(context, 'Premium Upgrade');
  }

  void _handleDeviceConnectivityTap(BuildContext context) {
    _showComingSoonSnackbar(context, 'Device Connectivity');
  }

  void _handlePrivacyTap(BuildContext context) {
    _showComingSoonSnackbar(context, 'Privacy & LOD');
  }

  void _handleExportDataTap(BuildContext context) {
    _showComingSoonSnackbar(context, 'Export Data');
  }

  void _handleThemeTap(BuildContext context) {
    showDialog(
      context: context,
      builder: (dialogContext) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: AppRadius.radiusLg),
        title: Text(
          'Select Theme',
          style: AppTypography.headingBoldStyle(
            fontSize: AppTypography.fontSizeLg.sp,
            color: AppColors.textPrimary,
          ),
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            _buildThemeOption(
              context,
              icon: Icons.light_mode,
              title: 'Light',
              themeMode: 'light',
            ),
            SizedBox(height: AppSpacing.xs),
            _buildThemeOption(
              context,
              icon: Icons.dark_mode,
              title: 'Dark',
              themeMode: 'dark',
            ),
            SizedBox(height: AppSpacing.xs),
            _buildThemeOption(
              context,
              icon: Icons.brightness_auto,
              title: 'System Default',
              themeMode: 'system',
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildThemeOption(
    BuildContext context, {
    required IconData icon,
    required String title,
    required String themeMode,
  }) {
    return InkWell(
      onTap: () {
        context.read<MoreBloc>().add(ToggleThemeMode(themeMode));
        Navigator.pop(context);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Theme selected: $title'),
            behavior: SnackBarBehavior.floating,
            shape: RoundedRectangleBorder(borderRadius: AppRadius.radiusMd),
            margin: AppSpacing.paddingMd,
          ),
        );
      },
      borderRadius: AppRadius.radiusMd,
      child: Container(
        padding: AppSpacing.paddingSm,
        decoration: BoxDecoration(
          border: Border.all(
            color: AppColors.primary.withOpacity(0.2),
            width: 1,
          ),
          borderRadius: AppRadius.radiusMd,
        ),
        child: Row(
          children: [
            Icon(icon, color: AppColors.primary, size: 24.sp),
            SizedBox(width: AppSpacing.sm),
            Text(
              title,
              style: AppTypography.bodyRegularStyle(
                fontSize: AppTypography.fontSizeBase.sp,
                color: AppColors.textPrimary,
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _handleLanguageTap(BuildContext context) {
    showDialog(
      context: context,
      builder: (dialogContext) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: AppRadius.radiusLg),
        title: Text(
          'Select Language',
          style: AppTypography.headingBoldStyle(
            fontSize: AppTypography.fontSizeLg.sp,
            color: AppColors.textPrimary,
          ),
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            _buildLanguageOption(context, 'Vietnamese', 'vi'),
            SizedBox(height: AppSpacing.xs),
            _buildLanguageOption(context, 'English', 'en'),
          ],
        ),
      ),
    );
  }

  Widget _buildLanguageOption(
    BuildContext context,
    String title,
    String languageCode,
  ) {
    return InkWell(
      onTap: () {
        context.read<MoreBloc>().add(ChangeLanguage(languageCode));
        Navigator.pop(context);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Language selected: $title'),
            behavior: SnackBarBehavior.floating,
            shape: RoundedRectangleBorder(borderRadius: AppRadius.radiusMd),
            margin: AppSpacing.paddingMd,
          ),
        );
      },
      borderRadius: AppRadius.radiusMd,
      child: Container(
        padding: AppSpacing.paddingSm,
        decoration: BoxDecoration(
          border: Border.all(
            color: AppColors.primary.withOpacity(0.2),
            width: 1,
          ),
          borderRadius: AppRadius.radiusMd,
        ),
        child: Row(
          children: [
            Icon(Icons.language, color: AppColors.primary, size: 24.sp),
            SizedBox(width: AppSpacing.sm),
            Text(
              title,
              style: AppTypography.bodyRegularStyle(
                fontSize: AppTypography.fontSizeBase.sp,
                color: AppColors.textPrimary,
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _handleNotificationsTap(BuildContext context, MoreLoaded state) {
    _showComingSoonSnackbar(context, 'Notification Settings');
  }

  void _handleRateUsTap(BuildContext context) {
    _showComingSoonSnackbar(context, 'Rate App');
  }

  void _handleFeedbackTap(BuildContext context) {
    _showComingSoonSnackbar(context, 'Send Feedback');
  }

  void _handleAboutTap(BuildContext context) {
    _showComingSoonSnackbar(context, 'About');
  }

  void _handleLogoutTap(BuildContext context) {
    showDialog(
      context: context,
      builder: (dialogContext) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: AppRadius.radiusLg),
        title: Text(
          'Logout',
          style: AppTypography.headingBoldStyle(
            fontSize: AppTypography.fontSizeLg.sp,
            color: AppColors.textPrimary,
          ),
        ),
        content: Text(
          'Are you sure you want to logout?',
          style: AppTypography.bodyRegularStyle(
            fontSize: AppTypography.fontSizeBase.sp,
            color: AppColors.textSecondary,
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(dialogContext),
            style: TextButton.styleFrom(
              padding: EdgeInsets.symmetric(
                horizontal: AppSpacing.lg,
                vertical: AppSpacing.sm,
              ),
            ),
            child: Text(
              'Cancel',
              style: AppTypography.bodyBoldStyle(
                fontSize: AppTypography.fontSizeBase.sp,
                color: AppColors.textSecondary,
              ),
            ),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.pop(dialogContext);
              context.read<AuthenticationBloc>().add(AuthenticationLoggedOut());
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: AppColors.danger,
              padding: EdgeInsets.symmetric(
                horizontal: AppSpacing.lg,
                vertical: AppSpacing.sm,
              ),
              shape: RoundedRectangleBorder(borderRadius: AppRadius.radiusMd),
            ),
            child: Text(
              'Logout',
              style: AppTypography.bodyBoldStyle(
                fontSize: AppTypography.fontSizeBase.sp,
                color: AppColors.white,
              ),
            ),
          ),
        ],
      ),
    );
  }

  void _showComingSoonSnackbar(BuildContext context, String feature) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('$feature - Coming soon!'),
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: AppRadius.radiusMd),
        margin: AppSpacing.paddingMd,
      ),
    );
  }
}

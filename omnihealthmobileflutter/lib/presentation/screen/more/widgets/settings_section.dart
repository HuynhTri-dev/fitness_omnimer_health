import 'package:flutter/material.dart';
import 'package:omnihealthmobileflutter/core/theme/app_spacing.dart';
import 'package:omnihealthmobileflutter/presentation/screen/more/bloc/more_state.dart';
import 'package:omnihealthmobileflutter/presentation/screen/more/widgets/more_menu_item.dart';

/// Settings section widget for More screen
class SettingsSection extends StatelessWidget {
  final MoreLoaded state;
  final VoidCallback onThemeTap;
  final VoidCallback onLanguageTap;
  final VoidCallback onNotificationsTap;

  const SettingsSection({
    Key? key,
    required this.state,
    required this.onThemeTap,
    required this.onLanguageTap,
    required this.onNotificationsTap,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        MoreMenuItem(
          icon: Icons.palette_outlined,
          title: 'Theme',
          subtitle: _getThemeLabel(state.themeMode),
          onTap: onThemeTap,
        ),
        SizedBox(height: AppSpacing.sm),
        MoreMenuItem(
          icon: Icons.language_outlined,
          title: 'Language',
          subtitle: state.languageCode == 'vi' ? 'Vietnamese' : 'English',
          onTap: onLanguageTap,
        ),
        SizedBox(height: AppSpacing.sm),
        MoreMenuItem(
          icon: Icons.notifications_outlined,
          title: 'Notifications',
          subtitle: 'Manage reminders and alerts',
          onTap: onNotificationsTap,
        ),
      ],
    );
  }

  String _getThemeLabel(String themeMode) {
    switch (themeMode) {
      case 'light':
        return 'Light';
      case 'dark':
        return 'Dark';
      case 'system':
      default:
        return 'System Default';
    }
  }
}

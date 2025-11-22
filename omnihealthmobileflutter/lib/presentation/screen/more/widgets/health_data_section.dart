import 'package:flutter/material.dart';
import 'package:omnihealthmobileflutter/core/theme/app_spacing.dart';
import 'package:omnihealthmobileflutter/presentation/screen/more/widgets/more_menu_item.dart';

/// Health & Data section widget for More screen
class HealthDataSection extends StatelessWidget {
  final VoidCallback onDeviceConnectivityTap;
  final VoidCallback onPrivacyTap;
  final VoidCallback onExportDataTap;

  const HealthDataSection({
    Key? key,
    required this.onDeviceConnectivityTap,
    required this.onPrivacyTap,
    required this.onExportDataTap,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        MoreMenuItem(
          icon: Icons.watch_outlined,
          title: 'Connect Devices',
          subtitle: 'Apple Watch, Samsung Watch, etc.',
          onTap: onDeviceConnectivityTap,
        ),
        SizedBox(height: AppSpacing.sm),
        MoreMenuItem(
          icon: Icons.lock_outline,
          title: 'Privacy & LOD',
          subtitle: 'Manage data sharing settings',
          onTap: onPrivacyTap,
        ),
        SizedBox(height: AppSpacing.sm),
        MoreMenuItem(
          icon: Icons.file_download_outlined,
          title: 'Export Data',
          subtitle: 'Download health reports (PDF/Excel)',
          onTap: onExportDataTap,
        ),
      ],
    );
  }
}

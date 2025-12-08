import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:omnihealthmobileflutter/core/theme/app_spacing.dart';
import 'package:omnihealthmobileflutter/presentation/screen/more/widgets/more_menu_item.dart';

import 'package:omnihealthmobileflutter/presentation/screen/health_connect/health_connect_screen.dart';
import 'package:omnihealthmobileflutter/presentation/screen/health_connect/health_connect_setup_widget.dart';
import 'package:omnihealthmobileflutter/presentation/screen/healthkit_connect/healthkit_connect_screen.dart';
import 'package:omnihealthmobileflutter/presentation/screen/healthkit_connect/healthkit_connect_setup_widget.dart';

/// Health & Data section widget for More screen
class HealthDataSection extends StatefulWidget {
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
  State<HealthDataSection> createState() => _HealthDataSectionState();
}

class _HealthDataSectionState extends State<HealthDataSection>
    with SingleTickerProviderStateMixin {
  bool _isExpanded = false;
  late AnimationController _animationController;
  late Animation<double> _expandAnimation;

  // State for health data sources

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
        // Health Data Center with Dropdown
        MoreMenuItem(
          icon: Icons.watch_outlined,
          title: 'Health Data Center',
          subtitle: 'Apple Health, Health Connect, Samsung Health',
          onTap: _toggleExpanded,
          trailing: Builder(
            builder: (context) => RotationTransition(
              turns: Tween(begin: 0.0, end: 0.5).animate(_expandAnimation),
              child: Icon(
                Icons.keyboard_arrow_down,
                color: Theme.of(
                  context,
                ).textTheme.bodySmall?.color?.withOpacity(0.6),
                size: 24.sp,
              ),
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
                  color: Theme.of(context).colorScheme.primary.withOpacity(0.3),
                  width: 2,
                ),
              ),
            ),
            child: Column(
              children: [
                Padding(
                  padding: EdgeInsets.symmetric(
                    horizontal: AppSpacing.md,
                    vertical: AppSpacing.xs,
                  ),
                  child: HealthKitConnectSetupWidget(
                    onNavigateToHealthKit: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => const HealthKitConnectScreen(),
                        ),
                      );
                    },
                  ),
                ),

                // Health Connect integration using HealthConnectSetupWidget
                Padding(
                  padding: EdgeInsets.symmetric(
                    horizontal: AppSpacing.md,
                    vertical: AppSpacing.xs,
                  ),
                  child: HealthConnectSetupWidget(
                    onNavigateToHealthConnect: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => const HealthConnectScreen(),
                        ),
                      );
                    },
                  ),
                ),
              ],
            ),
          ),
        ),

        SizedBox(height: AppSpacing.sm),

        // Privacy & LOD
        MoreMenuItem(
          icon: Icons.lock_outline,
          title: 'Privacy & LOD',
          subtitle: 'Manage data sharing settings',
          onTap: widget.onPrivacyTap,
        ),
        SizedBox(height: AppSpacing.sm),

        // Export Data
        MoreMenuItem(
          icon: Icons.file_download_outlined,
          title: 'Export Data',
          subtitle: 'Download health reports (PDF/Excel)',
          onTap: widget.onExportDataTap,
        ),
      ],
    );
  }
}

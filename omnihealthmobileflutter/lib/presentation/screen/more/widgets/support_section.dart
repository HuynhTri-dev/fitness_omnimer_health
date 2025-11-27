import 'package:flutter/material.dart';
import 'package:omnihealthmobileflutter/core/theme/app_spacing.dart';
import 'package:omnihealthmobileflutter/presentation/screen/more/widgets/more_menu_item.dart';

/// Support section widget for More screen
class SupportSection extends StatelessWidget {
  final VoidCallback onRateUsTap;
  final VoidCallback onFeedbackTap;
  final VoidCallback onAboutTap;

  const SupportSection({
    Key? key,
    required this.onRateUsTap,
    required this.onFeedbackTap,
    required this.onAboutTap,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        MoreMenuItem(
          icon: Icons.star_outline,
          title: 'Rate App',
          subtitle: 'Share your feedback with us',
          onTap: onRateUsTap,
        ),
        SizedBox(height: AppSpacing.sm),
        MoreMenuItem(
          icon: Icons.feedback_outlined,
          title: 'Send Feedback',
          subtitle: 'Report bugs or suggest features',
          onTap: onFeedbackTap,
        ),
        SizedBox(height: AppSpacing.sm),
        MoreMenuItem(
          icon: Icons.info_outline,
          title: 'About',
          subtitle: 'Version, terms, and privacy policy',
          onTap: onAboutTap,
        ),
      ],
    );
  }
}

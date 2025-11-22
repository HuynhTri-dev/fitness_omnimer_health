import 'package:equatable/equatable.dart';

/// Events cho More screen
abstract class MoreEvent extends Equatable {
  const MoreEvent();

  @override
  List<Object?> get props => [];
}

/// Event để toggle theme mode
class ToggleThemeMode extends MoreEvent {
  final String themeMode; // 'light', 'dark', 'system'

  const ToggleThemeMode(this.themeMode);

  @override
  List<Object?> get props => [themeMode];
}

/// Event để thay đổi ngôn ngữ
class ChangeLanguage extends MoreEvent {
  final String languageCode; // 'en', 'vi'

  const ChangeLanguage(this.languageCode);

  @override
  List<Object?> get props => [languageCode];
}

/// Event để toggle notification settings
class ToggleNotification extends MoreEvent {
  final String notificationType; // 'workout', 'goal', 'news'
  final bool enabled;

  const ToggleNotification(this.notificationType, this.enabled);

  @override
  List<Object?> get props => [notificationType, enabled];
}

/// Event để load settings
class LoadMoreSettings extends MoreEvent {
  const LoadMoreSettings();
}

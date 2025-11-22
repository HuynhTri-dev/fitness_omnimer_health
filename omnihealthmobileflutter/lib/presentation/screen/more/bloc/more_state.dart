import 'package:equatable/equatable.dart';

/// States cho More screen
abstract class MoreState extends Equatable {
  const MoreState();

  @override
  List<Object?> get props => [];
}

/// Initial state
class MoreInitial extends MoreState {
  const MoreInitial();
}

/// Loading state
class MoreLoading extends MoreState {
  const MoreLoading();
}

/// Loaded state với settings
class MoreLoaded extends MoreState {
  final String themeMode; // 'light', 'dark', 'system'
  final String languageCode; // 'en', 'vi'
  final Map<String, bool> notificationSettings;

  const MoreLoaded({
    required this.themeMode,
    required this.languageCode,
    required this.notificationSettings,
  });

  @override
  List<Object?> get props => [themeMode, languageCode, notificationSettings];

  MoreLoaded copyWith({
    String? themeMode,
    String? languageCode,
    Map<String, bool>? notificationSettings,
  }) {
    return MoreLoaded(
      themeMode: themeMode ?? this.themeMode,
      languageCode: languageCode ?? this.languageCode,
      notificationSettings: notificationSettings ?? this.notificationSettings,
    );
  }
}

/// Error state
class MoreError extends MoreState {
  final String message;

  const MoreError(this.message);

  @override
  List<Object?> get props => [message];
}

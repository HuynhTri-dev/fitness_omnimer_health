import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:omnihealthmobileflutter/presentation/screen/more/bloc/more_event.dart';
import 'package:omnihealthmobileflutter/presentation/screen/more/bloc/more_state.dart';

/// Bloc quản lý state cho More screen
class MoreBloc extends Bloc<MoreEvent, MoreState> {
  MoreBloc() : super(const MoreInitial()) {
    on<LoadMoreSettings>(_onLoadMoreSettings);
    on<ToggleThemeMode>(_onToggleThemeMode);
    on<ChangeLanguage>(_onChangeLanguage);
    on<ToggleNotification>(_onToggleNotification);
  }

  Future<void> _onLoadMoreSettings(
    LoadMoreSettings event,
    Emitter<MoreState> emit,
  ) async {
    emit(const MoreLoading());

    try {
      // TODO: Load settings from local storage (SharedPreferences/SecureStorage)
      // For now, use default values
      await Future.delayed(const Duration(milliseconds: 500));

      emit(
        const MoreLoaded(
          themeMode: 'system',
          languageCode: 'vi',
          notificationSettings: {'workout': true, 'goal': true, 'news': false},
        ),
      );
    } catch (e) {
      emit(MoreError(e.toString()));
    }
  }

  Future<void> _onToggleThemeMode(
    ToggleThemeMode event,
    Emitter<MoreState> emit,
  ) async {
    if (state is MoreLoaded) {
      final currentState = state as MoreLoaded;

      // TODO: Save to local storage
      emit(currentState.copyWith(themeMode: event.themeMode));
    }
  }

  Future<void> _onChangeLanguage(
    ChangeLanguage event,
    Emitter<MoreState> emit,
  ) async {
    if (state is MoreLoaded) {
      final currentState = state as MoreLoaded;

      // TODO: Save to local storage and update app locale
      emit(currentState.copyWith(languageCode: event.languageCode));
    }
  }

  Future<void> _onToggleNotification(
    ToggleNotification event,
    Emitter<MoreState> emit,
  ) async {
    if (state is MoreLoaded) {
      final currentState = state as MoreLoaded;
      final updatedSettings = Map<String, bool>.from(
        currentState.notificationSettings,
      );
      updatedSettings[event.notificationType] = event.enabled;

      // TODO: Save to local storage
      emit(currentState.copyWith(notificationSettings: updatedSettings));
    }
  }
}

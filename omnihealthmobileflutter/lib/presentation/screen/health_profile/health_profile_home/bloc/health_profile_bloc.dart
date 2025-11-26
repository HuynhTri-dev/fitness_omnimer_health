import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:omnihealthmobileflutter/domain/usecases/base_usecase.dart';
import 'package:omnihealthmobileflutter/domain/usecases/health_profile/create_health_profile.dart';
import 'package:omnihealthmobileflutter/domain/usecases/health_profile/delete_health_profile.dart';
import 'package:omnihealthmobileflutter/domain/usecases/health_profile/get_health_profile_by_id.dart';
import 'package:omnihealthmobileflutter/domain/usecases/health_profile/get_health_profiles.dart';
import 'package:omnihealthmobileflutter/domain/usecases/health_profile/get_health_profiles_by_user_id.dart';
import 'package:omnihealthmobileflutter/domain/usecases/health_profile/get_latest_health_profile.dart';
import 'package:omnihealthmobileflutter/domain/usecases/health_profile/update_health_profile.dart';
import 'health_profile_event.dart';
import 'health_profile_state.dart';

class HealthProfileBloc extends Bloc<HealthProfileEvent, HealthProfileState> {
  final GetHealthProfilesUseCase _getHealthProfilesUseCase;
  final GetHealthProfileByIdUseCase _getHealthProfileByIdUseCase;
  final GetLatestHealthProfileUseCase _getLatestHealthProfileUseCase;
  final GetHealthProfilesByUserIdUseCase _getHealthProfilesByUserIdUseCase;
  final CreateHealthProfileUseCase _createHealthProfileUseCase;
  final UpdateHealthProfileUseCase _updateHealthProfileUseCase;
  final DeleteHealthProfileUseCase _deleteHealthProfileUseCase;

  HealthProfileBloc({
    required GetHealthProfilesUseCase getHealthProfilesUseCase,
    required GetHealthProfileByIdUseCase getHealthProfileByIdUseCase,
    required GetLatestHealthProfileUseCase getLatestHealthProfileUseCase,
    required GetHealthProfilesByUserIdUseCase getHealthProfilesByUserIdUseCase,
    required CreateHealthProfileUseCase createHealthProfileUseCase,
    required UpdateHealthProfileUseCase updateHealthProfileUseCase,
    required DeleteHealthProfileUseCase deleteHealthProfileUseCase,
  }) : _getHealthProfilesUseCase = getHealthProfilesUseCase,
       _getHealthProfileByIdUseCase = getHealthProfileByIdUseCase,
       _getLatestHealthProfileUseCase = getLatestHealthProfileUseCase,
       _getHealthProfilesByUserIdUseCase = getHealthProfilesByUserIdUseCase,
       _createHealthProfileUseCase = createHealthProfileUseCase,
       _updateHealthProfileUseCase = updateHealthProfileUseCase,
       _deleteHealthProfileUseCase = deleteHealthProfileUseCase,
       super(const HealthProfileInitial()) {
    on<GetHealthProfilesEvent>(_onGetHealthProfiles);
    on<GetHealthProfileByIdEvent>(_onGetHealthProfileById);
    on<GetLatestHealthProfileEvent>(_onGetLatestHealthProfile);
    on<GetHealthProfilesByUserIdEvent>(_onGetHealthProfilesByUserId);
    on<CreateHealthProfileEvent>(_onCreateHealthProfile);
    on<UpdateHealthProfileEvent>(_onUpdateHealthProfile);
    on<DeleteHealthProfileEvent>(_onDeleteHealthProfile);
  }

  Future<void> _onGetHealthProfiles(
    GetHealthProfilesEvent event,
    Emitter<HealthProfileState> emit,
  ) async {
    emit(const HealthProfileLoading());
    try {
      final response = await _getHealthProfilesUseCase(NoParams());
      if (response.success && response.data != null) {
        emit(HealthProfilesLoaded(response.data!));
      } else {
        emit(HealthProfileError(response.message));
      }
    } catch (e) {
      emit(HealthProfileError(_extractErrorMessage(e)));
    }
  }

  Future<void> _onGetHealthProfileById(
    GetHealthProfileByIdEvent event,
    Emitter<HealthProfileState> emit,
  ) async {
    emit(const HealthProfileLoading());
    try {
      final response = await _getHealthProfileByIdUseCase(event.id);
      if (response.success && response.data != null) {
        emit(HealthProfileLoaded(response.data!));
      } else {
        emit(HealthProfileError(response.message));
      }
    } catch (e) {
      emit(HealthProfileError(_extractErrorMessage(e)));
    }
  }

  Future<void> _onGetLatestHealthProfile(
    GetLatestHealthProfileEvent event,
    Emitter<HealthProfileState> emit,
  ) async {
    emit(const HealthProfileLoading());
    try {
      final response = await _getLatestHealthProfileUseCase(NoParams());
      if (response.success && response.data != null) {
        emit(HealthProfileLoaded(response.data!));
      } else {
        emit(HealthProfileError(response.message));
      }
    } catch (e) {
      emit(HealthProfileError(_extractErrorMessage(e)));
    }
  }

  Future<void> _onGetHealthProfilesByUserId(
    GetHealthProfilesByUserIdEvent event,
    Emitter<HealthProfileState> emit,
  ) async {
    emit(const HealthProfileLoading());
    try {
      final response = await _getHealthProfilesByUserIdUseCase(event.userId);
      if (response.success && response.data != null) {
        emit(HealthProfilesLoaded(response.data!));
      } else {
        emit(HealthProfileError(response.message));
      }
    } catch (e) {
      emit(HealthProfileError(_extractErrorMessage(e)));
    }
  }

  Future<void> _onCreateHealthProfile(
    CreateHealthProfileEvent event,
    Emitter<HealthProfileState> emit,
  ) async {
    emit(const HealthProfileLoading());
    try {
      final response = await _createHealthProfileUseCase(event.profile);
      if (response.success && response.data != null) {
        emit(HealthProfileCreateSuccess(response.data!));
      } else {
        emit(HealthProfileError(response.message));
      }
    } catch (e) {
      emit(HealthProfileError(_extractErrorMessage(e)));
    }
  }

  Future<void> _onUpdateHealthProfile(
    UpdateHealthProfileEvent event,
    Emitter<HealthProfileState> emit,
  ) async {
    emit(const HealthProfileLoading());
    try {
      final response = await _updateHealthProfileUseCase(
        UpdateHealthProfileParams(id: event.id, profile: event.profile),
      );
      if (response.success && response.data != null) {
        emit(HealthProfileUpdateSuccess(response.data!));
      } else {
        emit(HealthProfileError(response.message));
      }
    } catch (e) {
      emit(HealthProfileError(_extractErrorMessage(e)));
    }
  }

  Future<void> _onDeleteHealthProfile(
    DeleteHealthProfileEvent event,
    Emitter<HealthProfileState> emit,
  ) async {
    emit(const HealthProfileLoading());
    try {
      final response = await _deleteHealthProfileUseCase(event.id);
      if (response.success) {
        emit(const HealthProfileDeleteSuccess());
      } else {
        emit(HealthProfileError(response.message));
      }
    } catch (e) {
      emit(HealthProfileError(_extractErrorMessage(e)));
    }
  }

  String _extractErrorMessage(dynamic error) {
    if (error.toString().contains('Exception:')) {
      return error.toString().replaceAll('Exception:', '').trim();
    }
    return error.toString();
  }
}

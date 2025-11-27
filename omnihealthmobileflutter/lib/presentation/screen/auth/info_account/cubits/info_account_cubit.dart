import 'dart:convert';
import 'dart:io';

import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:omnihealthmobileflutter/core/constants/enum_constant.dart';
import 'package:omnihealthmobileflutter/core/constants/storage_constant.dart';
import 'package:omnihealthmobileflutter/data/models/auth/user_model.dart';
import 'package:omnihealthmobileflutter/domain/entities/auth/auth_entity.dart';
import 'package:omnihealthmobileflutter/domain/entities/auth/user_entity.dart';
import 'package:omnihealthmobileflutter/domain/usecases/auth/update_user_usecase.dart';
import 'package:omnihealthmobileflutter/presentation/common/blocs/auth/authentication_bloc.dart';
import 'package:omnihealthmobileflutter/presentation/common/blocs/auth/authentication_event.dart';
import 'package:omnihealthmobileflutter/presentation/screen/auth/info_account/cubits/info_account_state.dart';
import 'package:omnihealthmobileflutter/services/shared_preferences_service.dart';

class InfoAccountCubit extends Cubit<InfoAccountState> {
  final SharedPreferencesService _sharedPreferencesService;
  final UpdateUserUseCase _updateUserUseCase;
  final AuthenticationBloc _authenticationBloc;

  InfoAccountCubit({
    required SharedPreferencesService sharedPreferencesService,
    required UpdateUserUseCase updateUserUseCase,
    required AuthenticationBloc authenticationBloc,
  }) : _sharedPreferencesService = sharedPreferencesService,
       _updateUserUseCase = updateUserUseCase,
       _authenticationBloc = authenticationBloc,
       super(InfoAccountInitial());

  Future<void> loadUserInfo() async {
    emit(InfoAccountLoading());
    try {
      final userJson = await _sharedPreferencesService.get<String>(
        StorageConstant.kUserInfoKey,
      );

      if (userJson != null) {
        final userMap = jsonDecode(userJson);
        final userModel = UserModel.fromJson(userMap);
        emit(InfoAccountLoaded(userModel.toEntity()));
      } else {
        emit(const InfoAccountError("Không tìm thấy thông tin người dùng"));
      }
    } catch (e) {
      emit(InfoAccountError("Lỗi tải thông tin: ${e.toString()}"));
    }
  }

  Future<void> updateUserInfo({
    required String id,
    String? fullname,
    String? birthday,
    GenderEnum? gender,
    String? imageUrl,
    File? image,
  }) async {
    emit(InfoAccountUpdating());

    // Create a UserEntity with ONLY the allowed fields
    final userToUpdate = UserEntity(
      fullname: fullname,
      birthday: birthday,
      gender: gender,
      imageUrl: imageUrl,
      image: image,
      // Forbidden fields are null
    );

    final result = await _updateUserUseCase.call(
      UpdateUserParams(id: id, user: userToUpdate),
    );

    if (result.success && result.data != null) {
      final updatedUser = result.data!;

      // Update Local Storage
      final userModel = UserModel.fromEntity(updatedUser);
      await _sharedPreferencesService.create(
        StorageConstant.kUserInfoKey,
        jsonEncode(userModel.toJson()),
      );

      // Notify AuthenticationBloc
      // Convert UserEntity to UserAuth
      // Note: UserEntity birthday is String, UserAuth birthday is DateTime
      DateTime? birthdayDt;
      if (updatedUser.birthday != null) {
        try {
          birthdayDt = DateTime.parse(updatedUser.birthday!);
        } catch (_) {}
      }

      final userAuth = UserAuth(
        id: updatedUser.id ?? id,
        fullname: updatedUser.fullname ?? "",
        email: updatedUser.email,
        imageUrl: updatedUser.imageUrl,
        gender: updatedUser.gender,
        birthday: birthdayDt,
        roleName: updatedUser.roleNames ?? [],
      );

      _authenticationBloc.add(AuthenticationUserUpdated(userAuth));

      emit(InfoAccountSuccess(updatedUser));
    } else {
      emit(InfoAccountError(result.message ?? "Cập nhật thất bại"));
      // Reload old data to reset UI state
      loadUserInfo();
    }
  }
}

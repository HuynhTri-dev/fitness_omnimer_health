import 'package:omnihealthmobileflutter/domain/entities/user_entity.dart';

class UserModel {
  final String email;
  final String? displayName;
  final String? photoUrl;
  final String? phoneNumber;
  final String? accessToken;
  final String? refreshToken;

  UserModel({
    required this.email,
    this.displayName,
    this.photoUrl,
    this.phoneNumber,
    this.accessToken,
    this.refreshToken,
  });

  factory UserModel.fromJson(Map<String, dynamic> json) {
    return UserModel(
      email: json['email'] as String,
      displayName: json['display_name'] as String?,
      photoUrl: json['photo_url'] as String?,
      phoneNumber: json['phone_number'] as String?,
      accessToken: json['access_token'] as String?,
      refreshToken: json['refresh_token'] as String?,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'email': email,
      'display_name': displayName,
      'photo_url': photoUrl,
      'phone_number': phoneNumber,
      'access_token': accessToken,
      'refresh_token': refreshToken,
    };
  }

  // Convert to Entity
  UserEntity toEntity() {
    return UserEntity(
      email: email,
      displayName: displayName,
      photoUrl: photoUrl,
      phoneNumber: phoneNumber,
    );
  }
}

class LoginRequestModel {
  final String email;
  final String password;

  LoginRequestModel({required this.email, required this.password});

  Map<String, dynamic> toJson() {
    return {'email': email, 'password': password};
  }

  // Convert from Entity
  factory LoginRequestModel.fromEntity(LoginEntity entity) {
    return LoginRequestModel(email: entity.email, password: entity.password);
  }
}

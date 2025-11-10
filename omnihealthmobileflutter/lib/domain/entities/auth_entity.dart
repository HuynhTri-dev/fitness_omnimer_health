import 'package:equatable/equatable.dart';

class LoginEntity extends Equatable {
  final String email;
  final String password;

  const LoginEntity({required this.email, required this.password});

  @override
  List<Object?> get props => [email, password];
}

class RegisterEntity extends Equatable {
  final String? email;
  final String? password;
  final String? fullname;
  final String? birthday;
  final String? gender;
  final List<String>? roleIds;
  final String? imageUrl;

  const RegisterEntity({
    this.email,
    this.password,
    this.fullname,
    this.birthday,
    this.gender,
    this.roleIds,
    this.imageUrl,
  });

  @override
  List<Object?> get props => [
    password,
    email,
    fullname,
    birthday,
    gender,
    roleIds,
  ];
}

import 'dart:io';

import 'package:equatable/equatable.dart';
import 'package:omnihealthmobileflutter/core/constants/enum_constant.dart';

class UserEntity extends Equatable {
  final String? email;
  final String? fullname;
  final String? birthday;
  final GenderEnum? gender;
  final List<String>? roleNames;
  final String? imageUrl;
  final File? image;

  const UserEntity({
    this.email,
    this.fullname,
    this.birthday,
    this.gender,
    this.roleNames,
    this.imageUrl,
    this.image,
  });

  @override
  List<Object?> get props => [
    email,
    fullname,
    birthday,
    gender,
    roleNames,
    imageUrl,
    image,
  ];
}

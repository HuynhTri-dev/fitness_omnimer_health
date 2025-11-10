import 'package:equatable/equatable.dart';

class UserEntity extends Equatable {
  final String? uid;
  final String? email;
  final String? fullname;
  final String? birthday;
  final String? gender;
  final List<String>? roleIds;
  final String? imageUrl;

  const UserEntity({
    this.uid,
    this.email,
    this.fullname,
    this.birthday,
    this.gender,
    this.roleIds,
    this.imageUrl,
  });

  @override
  List<Object?> get props => [uid, email, fullname, birthday, gender, roleIds];
}

import 'dart:io';

import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:omnihealthmobileflutter/utils/logger.dart';

/// Helper class để thao tác với Firebase Storage.
/// Mục tiêu:
/// - Tạo URL từ path lưu trong DB
/// - Không upload trực tiếp từ client (upload qua backend)
class FirebaseStorageUploader {
  final FirebaseStorage _storage;

  FirebaseStorageUploader({FirebaseStorage? storage})
    : _storage = storage ?? FirebaseStorage.instance;

  /// Lấy URL công khai từ 1 đường dẫn file trong Firebase Storage
  /// [path] ví dụ: `avatar_user/abc123.jpg`
  Future<String?> getDownloadUrlFromPath(String? path) async {
    if (path == null || path.isEmpty) return null;
    try {
      final ref = _storage.ref().child(path);
      return await ref.getDownloadURL();
    } catch (e) {
      return null;
    }
  }

  Future<Map<String, String>?> uploadAvatar(File imageFile) async {
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) return null;

    try {
      final avatarPath = 'avatar_user/${user.uid}';

      final ref = _storage.ref(avatarPath);
      await ref.putFile(imageFile);
      final avatarUrl = await ref.getDownloadURL();
      return {'url': avatarUrl, 'path': avatarPath};
    } catch (e) {
      logger.e('Upload avatar thất bại: $e');
      return null;
    }
  }
}

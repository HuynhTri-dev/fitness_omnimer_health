import 'app_config.dart';

class Endpoints {
  static String get baseUrl => AppConfig.baseUrl;

  // ================== AUTH ==================
  static const String login = "/v1/auth/login";
  static const String register = "/v1/auth/register";
  static const String logout = "/v1/auth/logout";
  static const String changePassword = "/v1/auth/change-password";
  static const String forgetPassword = "/v1/auth/forget-password";
}

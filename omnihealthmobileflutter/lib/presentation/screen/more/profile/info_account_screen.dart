import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:omnihealthmobileflutter/core/theme/app_colors.dart';
import 'package:omnihealthmobileflutter/core/theme/app_spacing.dart';
import 'package:omnihealthmobileflutter/core/theme/app_typography.dart';
import 'package:omnihealthmobileflutter/core/theme/app_radius.dart';

/// Info Account Screen - Update basic user information
/// Based on User model: fullname, email, birthday, gender, imageUrl
class InfoAccountScreen extends StatefulWidget {
  const InfoAccountScreen({Key? key}) : super(key: key);

  @override
  State<InfoAccountScreen> createState() => _InfoAccountScreenState();
}

class _InfoAccountScreenState extends State<InfoAccountScreen> {
  final _formKey = GlobalKey<FormState>();
  final TextEditingController _fullnameController = TextEditingController();
  final TextEditingController _emailController = TextEditingController();
  DateTime? _selectedBirthday;
  String _selectedGender = 'Other';
  String? _imageUrl;

  @override
  void initState() {
    super.initState();
    _loadUserData();
  }

  @override
  void dispose() {
    _fullnameController.dispose();
    _emailController.dispose();
    super.dispose();
  }

  void _loadUserData() {
    // TODO: Load user data from BLoC or repository
    // For now, using placeholder data
    _fullnameController.text = 'John Doe';
    _emailController.text = 'john.doe@example.com';
    _selectedBirthday = DateTime(1990, 1, 1);
    _selectedGender = 'Male';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        backgroundColor: AppColors.background,
        elevation: 0,
        leading: IconButton(
          icon: Icon(Icons.arrow_back_ios, color: AppColors.textPrimary),
          onPressed: () => Navigator.pop(context),
        ),
        title: Text(
          'Account Information',
          style: AppTypography.headingBoldStyle(
            fontSize: AppTypography.fontSizeLg.sp,
            color: AppColors.textPrimary,
          ),
        ),
        actions: [
          TextButton(
            onPressed: _saveChanges,
            child: Text(
              'Save',
              style: AppTypography.bodyBoldStyle(
                fontSize: AppTypography.fontSizeBase.sp,
                color: AppColors.primary,
              ),
            ),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: AppSpacing.paddingMd,
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Profile Image Section
              _buildProfileImageSection(),

              SizedBox(height: AppSpacing.lg),

              // Full Name Field
              _buildTextField(
                controller: _fullnameController,
                label: 'Full Name',
                hint: 'Enter your full name',
                icon: Icons.person_outline,
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Please enter your full name';
                  }
                  return null;
                },
              ),

              SizedBox(height: AppSpacing.md),

              // Email Field (Read-only, can be changed in Verify Account)
              _buildTextField(
                controller: _emailController,
                label: 'Email',
                hint: 'Your email address',
                icon: Icons.email_outlined,
                enabled: false,
                suffixIcon: IconButton(
                  icon: Icon(Icons.info_outline, color: AppColors.textMuted),
                  onPressed: () {
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(
                        content: const Text(
                          'To change email, go to Verify Account',
                        ),
                        behavior: SnackBarBehavior.floating,
                        shape: RoundedRectangleBorder(
                          borderRadius: AppRadius.radiusMd,
                        ),
                        margin: AppSpacing.paddingMd,
                      ),
                    );
                  },
                ),
              ),

              SizedBox(height: AppSpacing.md),

              // Birthday Field
              _buildBirthdayField(),

              SizedBox(height: AppSpacing.md),

              // Gender Field
              _buildGenderField(),

              SizedBox(height: AppSpacing.xl),

              // Save Button
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: _saveChanges,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.primary,
                    padding: EdgeInsets.symmetric(vertical: AppSpacing.md),
                    shape: RoundedRectangleBorder(
                      borderRadius: AppRadius.radiusMd,
                    ),
                  ),
                  child: Text(
                    'Save Changes',
                    style: AppTypography.bodyBoldStyle(
                      fontSize: AppTypography.fontSizeBase.sp,
                      color: AppColors.white,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildProfileImageSection() {
    return Center(
      child: Stack(
        children: [
          Container(
            width: 120.w,
            height: 120.w,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: AppColors.primary.withOpacity(0.1),
              image: _imageUrl != null
                  ? DecorationImage(
                      image: NetworkImage(_imageUrl!),
                      fit: BoxFit.cover,
                    )
                  : null,
            ),
            child: _imageUrl == null
                ? Icon(Icons.person, size: 60.sp, color: AppColors.primary)
                : null,
          ),
          Positioned(
            bottom: 0,
            right: 0,
            child: GestureDetector(
              onTap: _changeProfileImage,
              child: Container(
                padding: EdgeInsets.all(AppSpacing.xs),
                decoration: BoxDecoration(
                  color: AppColors.primary,
                  shape: BoxShape.circle,
                  border: Border.all(color: AppColors.white, width: 2),
                ),
                child: Icon(
                  Icons.camera_alt,
                  color: AppColors.white,
                  size: 20.sp,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTextField({
    required TextEditingController controller,
    required String label,
    required String hint,
    required IconData icon,
    bool enabled = true,
    Widget? suffixIcon,
    String? Function(String?)? validator,
  }) {
    return TextFormField(
      controller: controller,
      enabled: enabled,
      validator: validator,
      decoration: InputDecoration(
        labelText: label,
        hintText: hint,
        prefixIcon: Icon(icon, color: AppColors.primary),
        suffixIcon: suffixIcon,
        border: OutlineInputBorder(
          borderRadius: AppRadius.radiusMd,
          borderSide: BorderSide(color: AppColors.border),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: AppRadius.radiusMd,
          borderSide: BorderSide(color: AppColors.border),
        ),
        disabledBorder: OutlineInputBorder(
          borderRadius: AppRadius.radiusMd,
          borderSide: BorderSide(color: AppColors.border.withOpacity(0.5)),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: AppRadius.radiusMd,
          borderSide: BorderSide(color: AppColors.primary, width: 2),
        ),
        errorBorder: OutlineInputBorder(
          borderRadius: AppRadius.radiusMd,
          borderSide: BorderSide(color: AppColors.error),
        ),
        focusedErrorBorder: OutlineInputBorder(
          borderRadius: AppRadius.radiusMd,
          borderSide: BorderSide(color: AppColors.error, width: 2),
        ),
      ),
    );
  }

  Widget _buildBirthdayField() {
    return InkWell(
      onTap: _selectBirthday,
      child: Container(
        padding: EdgeInsets.symmetric(
          horizontal: AppSpacing.md,
          vertical: AppSpacing.sm + 4.h,
        ),
        decoration: BoxDecoration(
          border: Border.all(color: AppColors.border),
          borderRadius: AppRadius.radiusMd,
        ),
        child: Row(
          children: [
            Icon(Icons.cake_outlined, color: AppColors.primary),
            SizedBox(width: AppSpacing.md),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Birthday',
                    style: AppTypography.bodyRegularStyle(
                      fontSize: AppTypography.fontSizeSm.sp,
                      color: AppColors.textSecondary,
                    ),
                  ),
                  SizedBox(height: 4.h),
                  Text(
                    _selectedBirthday != null
                        ? '${_selectedBirthday!.day}/${_selectedBirthday!.month}/${_selectedBirthday!.year}'
                        : 'Select your birthday',
                    style: AppTypography.bodyRegularStyle(
                      fontSize: AppTypography.fontSizeBase.sp,
                      color: _selectedBirthday != null
                          ? AppColors.textPrimary
                          : AppColors.textMuted,
                    ),
                  ),
                ],
              ),
            ),
            Icon(Icons.calendar_today, color: AppColors.textMuted, size: 20.sp),
          ],
        ),
      ),
    );
  }

  Widget _buildGenderField() {
    return Container(
      padding: EdgeInsets.symmetric(
        horizontal: AppSpacing.md,
        vertical: AppSpacing.xs,
      ),
      decoration: BoxDecoration(
        border: Border.all(color: AppColors.border),
        borderRadius: AppRadius.radiusMd,
      ),
      child: Row(
        children: [
          Icon(Icons.wc_outlined, color: AppColors.primary),
          SizedBox(width: AppSpacing.md),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Gender',
                  style: AppTypography.bodyRegularStyle(
                    fontSize: AppTypography.fontSizeSm.sp,
                    color: AppColors.textSecondary,
                  ),
                ),
                DropdownButton<String>(
                  value: _selectedGender,
                  isExpanded: true,
                  underline: const SizedBox(),
                  items: ['Male', 'Female', 'Other'].map((String value) {
                    return DropdownMenuItem<String>(
                      value: value,
                      child: Text(
                        value,
                        style: AppTypography.bodyRegularStyle(
                          fontSize: AppTypography.fontSizeBase.sp,
                          color: AppColors.textPrimary,
                        ),
                      ),
                    );
                  }).toList(),
                  onChanged: (String? newValue) {
                    if (newValue != null) {
                      setState(() {
                        _selectedGender = newValue;
                      });
                    }
                  },
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Future<void> _selectBirthday() async {
    final DateTime? picked = await showDatePicker(
      context: context,
      initialDate: _selectedBirthday ?? DateTime.now(),
      firstDate: DateTime(1900),
      lastDate: DateTime.now(),
      builder: (context, child) {
        return Theme(
          data: Theme.of(context).copyWith(
            colorScheme: ColorScheme.light(
              primary: AppColors.primary,
              onPrimary: AppColors.white,
              onSurface: AppColors.textPrimary,
            ),
          ),
          child: child!,
        );
      },
    );

    if (picked != null && picked != _selectedBirthday) {
      setState(() {
        _selectedBirthday = picked;
      });
    }
  }

  void _changeProfileImage() {
    // TODO: Implement image picker
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: const Text('Image picker - Coming soon!'),
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: AppRadius.radiusMd),
        margin: AppSpacing.paddingMd,
      ),
    );
  }

  void _saveChanges() {
    if (_formKey.currentState!.validate()) {
      // TODO: Implement save changes to backend
      // Collect data:
      // - fullname: _fullnameController.text
      // - birthday: _selectedBirthday
      // - gender: _selectedGender
      // - imageUrl: _imageUrl

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: const Text('Changes saved successfully!'),
          behavior: SnackBarBehavior.floating,
          shape: RoundedRectangleBorder(borderRadius: AppRadius.radiusMd),
          margin: AppSpacing.paddingMd,
          backgroundColor: AppColors.success,
        ),
      );
    }
  }
}

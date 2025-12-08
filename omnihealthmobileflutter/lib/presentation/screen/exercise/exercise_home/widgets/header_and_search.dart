// lib/presentation/screen/exercise/exercise_home/widgets/header_and_search.dart
part of '../exercise_home_screen.dart';

class _BodyAndMuscleHeader extends StatelessWidget {
  final MuscleEntity? selectedMuscle;

  const _BodyAndMuscleHeader({required this.selectedMuscle});

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // ======= MODEL 3D BÊN TRÁI =======
        Expanded(
          flex: 3,
          child: _Body3DView(
            onMuscleTap: (muscleName) {
              context.read<ExerciseHomeBloc>().add(
                SelectMuscleByName(muscleName),
              );
            },
          ),
        ),

        SizedBox(width: 12.w),

        // ======= CARD MUSCLE BÊN PHẢI =======
        Expanded(flex: 2, child: _MuscleDetailCard(muscle: selectedMuscle)),
      ],
    );
  }
}

class _Body3DView extends StatefulWidget {
  final ValueChanged<String> onMuscleTap;

  const _Body3DView({required this.onMuscleTap});

  @override
  State<_Body3DView> createState() => _Body3DViewState();
}

class _Body3DViewState extends State<_Body3DView> {
  // ignore: unused_field
  late Scene _scene;
  Object? _rootObject;
  final Map<String, vector.Vector3> _cachedCenters = {};

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 220.h,
      decoration: BoxDecoration(
        color: Theme.of(context).cardColor,
        borderRadius: AppRadius.radiusXl,
        boxShadow: [
          BoxShadow(
            color: AppColors.shadow,
            blurRadius: 16,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      alignment: Alignment.center,
      child: ClipRRect(
        borderRadius: AppRadius.radiusXl,
        child: Stack(
          children: [
            Cube(
              onSceneCreated: (Scene scene) {
                _scene = scene;
                scene.camera.zoom =
                    15; // Zoom out a bit more to fit everything? 20 was default.
                // Keep keeping 20 if user didn't ask to change, but user said "dựa theo cái này để sửa lại".
                // I will stick to existing camera values.
                scene.camera.zoom = 20;
                scene.camera.position.setValues(30, 2, 0);

                _rootObject = Object(fileName: 'assets/models/mesh.obj');
                _rootObject!.scale.setValues(0.32, 0.32, 0.32);
                scene.world.add(_rootObject!);
              },
            ),
            // Layer gesture detector để bắt sự kiện tap (click)
            Positioned.fill(
              child: GestureDetector(
                onTapUp: _handleTap,
                behavior: HitTestBehavior.translucent,
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _handleTap(TapUpDetails details) {
    if (_rootObject == null) return;

    // 1. Ensure centers are calculated
    if (_cachedCenters.isEmpty) {
      _calculateCenters();
    }

    if (_cachedCenters.isEmpty) return;

    final renderBox = context.findRenderObject() as RenderBox?;
    if (renderBox == null) return;
    final size = renderBox.size;
    final tapPos = details.localPosition;

    // 2. Prepare matrices
    // Model Matrix (Root Object Transform)
    // Note: _rootObject!.transform includes scale, rotation, position.
    final modelMatrix = _rootObject!.transform;

    // View & Projection Matrix
    // scene.camera exposes separate matrices
    final viewMatrix = _scene.camera.lookAtMatrix;
    final projMatrix = _scene.camera.projectionMatrix;

    // MVP = P * V * M
    final mvpMatrix = vector.Matrix4.identity();
    mvpMatrix.multiply(projMatrix);
    mvpMatrix.multiply(viewMatrix);
    mvpMatrix.multiply(modelMatrix);

    // 3. Find closest muscle
    String? closestMuscleName;
    double minDistance = double.infinity;

    for (final entry in _cachedCenters.entries) {
      final name = entry.key;
      final centerLocal = entry.value;

      // Local to World+Clip
      final centerList = vector.Vector4(
        centerLocal.x,
        centerLocal.y,
        centerLocal.z,
        1.0,
      );
      final clipPos = mvpMatrix.transform(centerList);

      // Perspective Division to get NDC (-1 to 1)
      if (clipPos.w == 0) continue; // Avoid division by zero
      final ndcX = clipPos.x / clipPos.w;
      final ndcY = clipPos.y / clipPos.w;

      // Check if point is behind camera (optional but good practice)
      if (clipPos.w <= 0) continue;

      // NDC to Screen Coordinates
      final screenX = (ndcX + 1) * size.width / 2;
      final screenY = (1 - ndcY) * size.height / 2; // Assuming GL convention

      final dist =
          (vector.Vector2(screenX, screenY) -
                  vector.Vector2(tapPos.dx, tapPos.dy))
              .length;

      // Threshold for click? E.g. 50 pixels? Or just min?
      if (dist < minDistance && dist < 100) {
        // 100px radius tolerance
        minDistance = dist;
        closestMuscleName = name;
      }
    }

    if (closestMuscleName != null) {
      // "lấy tên để sau đó chuyển lại dấu _ thành dấu cách"
      final formattedName = closestMuscleName.replaceAll('_', ' ');
      widget.onMuscleTap(formattedName);
    }
  }

  void _calculateCenters() {
    if (_rootObject == null) return;

    // Check root children
    for (final child in _rootObject!.children) {
      _processObject(child);
    }
    // Also check root itself if it has mesh
    // We access mesh safely assuming it works, or handle dynamic/nullable
    // Using dynamic to bypass strict checks if uncertain about package version API
    // but cleaner is to trust previous analysis: try catch or safe access
    try {
      final mesh = _rootObject?.mesh;
      // Check if mesh and vertices are valid
      if (mesh != null) {
        try {
          if (mesh.vertices.isNotEmpty) {
            _processObject(_rootObject!);
          }
        } catch (_) {}
      }
    } catch (_) {
      // Mesh might be null or getter failed
    }
  }

  void _processObject(Object obj) {
    // Ignore unnamed or weird objects if necessary
    // Try to access mesh.vertices
    // Since linter complains about nullable receiver, we use ?. where possible.
    // However Mesh is a property.

    // Safely access vertices if possible
    List<vector.Vector3>? vertices;
    try {
      vertices = obj.mesh.vertices;
    } catch (_) {
      // If mesh is null
    }

    if (vertices != null && vertices.isNotEmpty && obj.name!.isNotEmpty) {
      double sumX = 0;
      double sumY = 0;
      double sumZ = 0;
      for (final v in vertices) {
        sumX += v.x;
        sumY += v.y;
        sumZ += v.z;
      }
      final count = vertices.length;
      _cachedCenters[obj.name!] = vector.Vector3(
        sumX / count,
        sumY / count,
        sumZ / count,
      );
    }

    // Recurse if needed
    for (final child in obj.children) {
      _processObject(child);
    }
  }
}

class _MuscleDetailCard extends StatelessWidget {
  final MuscleEntity? muscle;

  const _MuscleDetailCard({required this.muscle});

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 220.h,
      padding: EdgeInsets.all(12.w),
      decoration: BoxDecoration(
        color: Theme.of(context).cardColor,
        borderRadius: AppRadius.radiusXl,
        boxShadow: [
          BoxShadow(
            color: AppColors.shadow,
            blurRadius: 16,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            muscle?.name ?? 'Muscle Name',
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: Theme.of(
              context,
            ).textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w700),
          ),
          SizedBox(height: 4.h),
          Text(
            muscle?.description ?? 'Description...',
            maxLines: 3,
            overflow: TextOverflow.ellipsis,
            style: Theme.of(context).textTheme.labelSmall?.copyWith(
              color: Theme.of(context).textTheme.bodySmall?.color,
            ),
          ),
          const Spacer(),
          ClipRRect(
            borderRadius: AppRadius.radiusLg,
            child: (muscle?.imageUrl?.isNotEmpty == true)
                ? Image.network(
                    muscle!.imageUrl!,
                    height: 90.h,
                    width: double.infinity,
                    fit: BoxFit.cover,
                  )
                : Container(
                    height: 90.h,
                    width: double.infinity,
                    color: Theme.of(context).scaffoldBackgroundColor,
                    alignment: Alignment.center,
                    child: const Icon(Icons.image),
                  ),
          ),
        ],
      ),
    );
  }
}

class _FilterButton extends StatelessWidget {
  final int resultCount;
  final VoidCallback onPressed;

  const _FilterButton({required this.resultCount, required this.onPressed});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onPressed,
      child: Container(
        padding: EdgeInsets.symmetric(horizontal: 10.w, vertical: 8.h),
        decoration: BoxDecoration(
          color: Theme.of(context).cardColor,
          borderRadius: AppRadius.radiusLg,
          boxShadow: [
            BoxShadow(
              color: AppColors.shadow,
              blurRadius: 12,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.filter_list, size: 20, color: AppColors.primary),
            SizedBox(height: 4.h),
            Text(
              '$resultCount',
              style: Theme.of(
                context,
              ).textTheme.labelSmall?.copyWith(fontWeight: FontWeight.w600),
            ),
          ],
        ),
      ),
    );
  }
}

class _SearchField extends StatelessWidget {
  final String value;
  final ValueChanged<String> onChanged;

  const _SearchField({required this.value, required this.onChanged});

  @override
  Widget build(BuildContext context) {
    return TextField(
      onChanged: onChanged,
      controller: TextEditingController(text: value)
        ..selection = TextSelection.collapsed(offset: value.length),
      decoration: InputDecoration(
        hintText: 'Tìm kiếm bài tập...',
        hintStyle: Theme.of(context).inputDecorationTheme.hintStyle,
        prefixIcon: const Icon(Icons.search),
        filled: true,
        fillColor: Theme.of(context).inputDecorationTheme.fillColor,
        contentPadding: EdgeInsets.symmetric(horizontal: 12.w, vertical: 8.h),
        border: OutlineInputBorder(
          borderRadius: AppRadius.radiusLg,
          borderSide:
              Theme.of(context).inputDecorationTheme.border?.borderSide ??
              BorderSide.none,
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: AppRadius.radiusLg,
          borderSide:
              Theme.of(
                context,
              ).inputDecorationTheme.enabledBorder?.borderSide ??
              BorderSide.none,
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: AppRadius.radiusLg,
          borderSide:
              Theme.of(
                context,
              ).inputDecorationTheme.focusedBorder?.borderSide ??
              BorderSide(color: Theme.of(context).primaryColor),
        ),
      ),
    );
  }
}

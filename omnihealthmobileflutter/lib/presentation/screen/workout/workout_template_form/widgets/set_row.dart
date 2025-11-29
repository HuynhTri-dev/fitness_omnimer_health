part of '../workout_template_form_screen.dart';

class _SetRow extends StatelessWidget {
  final WorkoutSetFormData set;
  final int exerciseIndex;
  final int setIndex;

  const _SetRow({
    required this.set,
    required this.exerciseIndex,
    required this.setIndex,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        // Set number badge
        Container(
          width: 36.w,
          height: 36.w,
          alignment: Alignment.center,
          decoration: BoxDecoration(
            color: Theme.of(context).primaryColor.withOpacity(0.1),
            borderRadius: BorderRadius.circular(8.r),
          ),
          child: Text(
            '${set.setOrder}',
            style: TextStyle(
              fontSize: 14.sp,
              fontWeight: FontWeight.bold,
              color: Theme.of(context).primaryColor,
            ),
          ),
        ),

        SizedBox(width: 8.w),

        // Weight input
        Expanded(
          child: _InputField(
            key: ValueKey('weight_${exerciseIndex}_$setIndex'),
            initialValue: set.weight?.toString() ?? '',
            hintText: '0',
            suffix: 'kg',
            onChanged: (value) {
              final weight = double.tryParse(value);
              context.read<WorkoutTemplateFormCubit>().updateSet(
                    exerciseIndex,
                    setIndex,
                    weight: weight,
                  );
            },
          ),
        ),

        SizedBox(width: 8.w),

        // Reps input
        Expanded(
          child: _InputField(
            key: ValueKey('reps_${exerciseIndex}_$setIndex'),
            initialValue: set.reps?.toString() ?? '',
            hintText: '0',
            suffix: 'reps',
            onChanged: (value) {
              final reps = int.tryParse(value);
              context.read<WorkoutTemplateFormCubit>().updateSet(
                    exerciseIndex,
                    setIndex,
                    reps: reps,
                  );
            },
          ),
        ),

        // More menu
        SizedBox(
          width: 32.w,
          child: PopupMenuButton<String>(
            icon: Icon(Icons.more_vert, size: 18.sp, color: Colors.grey[600]),
            padding: EdgeInsets.zero,
            onSelected: (value) {
              if (value == 'delete') {
                context
                    .read<WorkoutTemplateFormCubit>()
                    .removeSet(exerciseIndex, setIndex);
              }
            },
            itemBuilder: (context) => [
              const PopupMenuItem(
                value: 'delete',
                child: Row(
                  children: [
                    Icon(Icons.delete, size: 18, color: Colors.red),
                    SizedBox(width: 8),
                    Text('Delete', style: TextStyle(color: Colors.red)),
                  ],
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _InputField extends StatefulWidget {
  final String initialValue;
  final String hintText;
  final String? suffix;
  final ValueChanged<String> onChanged;

  const _InputField({
    super.key,
    required this.initialValue,
    required this.hintText,
    this.suffix,
    required this.onChanged,
  });

  @override
  State<_InputField> createState() => _InputFieldState();
}

class _InputFieldState extends State<_InputField> {
  late TextEditingController _controller;

  @override
  void initState() {
    super.initState();
    _controller = TextEditingController(text: widget.initialValue);
  }

  @override
  void didUpdateWidget(covariant _InputField oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.initialValue != widget.initialValue &&
        _controller.text != widget.initialValue) {
      _controller.text = widget.initialValue;
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 40.h,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(10.r),
        border: Border.all(
          color: Colors.grey.withOpacity(0.3),
          width: 1,
        ),
      ),
      child: Row(
        children: [
          Expanded(
            child: TextField(
              controller: _controller,
              textAlign: TextAlign.center,
              keyboardType: const TextInputType.numberWithOptions(decimal: true),
              decoration: InputDecoration(
                hintText: widget.hintText,
                hintStyle: TextStyle(
                  color: Colors.grey[400],
                  fontSize: 14.sp,
                ),
                border: InputBorder.none,
                isDense: true,
                contentPadding: EdgeInsets.symmetric(
                  horizontal: 8.w,
                  vertical: 10.h,
                ),
              ),
              style: TextStyle(
                fontSize: 14.sp,
                fontWeight: FontWeight.w600,
                color: Theme.of(context).textTheme.bodyLarge?.color,
              ),
              onChanged: widget.onChanged,
            ),
          ),
          if (widget.suffix != null) ...[
            Container(
              padding: EdgeInsets.symmetric(horizontal: 8.w),
              decoration: BoxDecoration(
                color: Theme.of(context).primaryColor.withOpacity(0.1),
                borderRadius: BorderRadius.only(
                  topRight: Radius.circular(9.r),
                  bottomRight: Radius.circular(9.r),
                ),
              ),
              height: double.infinity,
              alignment: Alignment.center,
              child: Text(
                widget.suffix!,
                style: TextStyle(
                  fontSize: 10.sp,
                  fontWeight: FontWeight.w600,
                  color: Theme.of(context).primaryColor,
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }
}


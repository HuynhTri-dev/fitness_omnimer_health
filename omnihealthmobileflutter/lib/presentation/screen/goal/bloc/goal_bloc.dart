import 'package:bloc/bloc.dart';
import 'package:omnihealthmobileflutter/domain/entities/goal_entity.dart';
import 'package:omnihealthmobileflutter/domain/usecases/goal/create_goal_usecase.dart';
import 'package:omnihealthmobileflutter/domain/usecases/goal/delete_goal_usecase.dart';
import 'package:omnihealthmobileflutter/domain/usecases/goal/get_goals_usecase.dart';
import 'package:omnihealthmobileflutter/domain/usecases/goal/update_goal_usecase.dart';
import 'goal_event.dart';
import 'goal_state.dart';

class GoalBloc extends Bloc<GoalEvent, GoalState> {
  final GetGoalsUseCase _getGoalsUseCase;
  final CreateGoalUseCase _createGoalUseCase;
  final UpdateGoalUseCase _updateGoalUseCase;
  final DeleteGoalUseCase _deleteGoalUseCase;

  GoalBloc({
    required GetGoalsUseCase getGoalsUseCase,
    required CreateGoalUseCase createGoalUseCase,
    required UpdateGoalUseCase updateGoalUseCase,
    required DeleteGoalUseCase deleteGoalUseCase,
  }) : _getGoalsUseCase = getGoalsUseCase,
       _createGoalUseCase = createGoalUseCase,
       _updateGoalUseCase = updateGoalUseCase,
       _deleteGoalUseCase = deleteGoalUseCase,
       super(GoalLoading()) {
    on<LoadGoalsEvent>(_onLoadGoals);
    on<CreateGoalEvent>(_onCreateGoal);
    on<UpdateGoalEvent>(_onUpdateGoal);
    on<DeleteGoalEvent>(_onDeleteGoal);
  }

  Future<void> _onLoadGoals(
    LoadGoalsEvent event,
    Emitter<GoalState> emit,
  ) async {
    emit(GoalLoading());
    try {
      final response = await _getGoalsUseCase(event.userId);
      if (response.success && response.data != null) {
        emit(GoalsLoaded(response.data!));
      } else {
        emit(GoalError(response.message));
      }
    } catch (e) {
      emit(GoalError('Không thể tải mục tiêu: ${e.toString()}'));
    }
  }

  Future<void> _onCreateGoal(
    CreateGoalEvent event,
    Emitter<GoalState> emit,
  ) async {
    emit(GoalLoading());
    try {
      final response = await _createGoalUseCase(event.goal);
      if (response.success && response.data != null) {
        emit(GoalOperationSuccess(response.data!));
        add(LoadGoalsEvent(event.goal.userId));
      } else {
        emit(GoalError(response.message));
      }
    } catch (e) {
      emit(GoalError('Tạo mục tiêu thất bại: ${e.toString()}'));
    }
  }

  Future<void> _onUpdateGoal(
    UpdateGoalEvent event,
    Emitter<GoalState> emit,
  ) async {
    emit(GoalLoading());
    try {
      final response = await _updateGoalUseCase(event.goal);
      if (response.success && response.data != null) {
        emit(GoalOperationSuccess(response.data!));
        add(LoadGoalsEvent(event.goal.userId));
      } else {
        emit(GoalError(response.message));
      }
    } catch (e) {
      emit(GoalError('Cập nhật mục tiêu thất bại: ${e.toString()}'));
    }
  }

  Future<void> _onDeleteGoal(
    DeleteGoalEvent event,
    Emitter<GoalState> emit,
  ) async {
    emit(GoalLoading());
    try {
      final response = await _deleteGoalUseCase(event.goalId);
      if (response.success) {
        emit(const GoalDeleted(noGoalsLeft: false));
      } else {
        emit(GoalError(response.message));
      }
    } catch (e) {
      emit(GoalError('Xóa mục tiêu thất bại: ${e.toString()}'));
    }
  }
}

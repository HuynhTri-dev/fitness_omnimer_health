import 'package:omnihealthmobileflutter/core/api/api_response.dart';
import 'package:omnihealthmobileflutter/domain/abstracts/muscle_repository_abs.dart';
import 'package:omnihealthmobileflutter/domain/entities/exercise/muscle_entity.dart';
import 'package:omnihealthmobileflutter/domain/usecases/base_usecase.dart';

class GetMuscleByNameUseCase
    implements UseCase<ApiResponse<MuscleEntity>, String> {
  final MuscleRepositoryAbs repository;

  GetMuscleByNameUseCase(this.repository);

  @override
  Future<ApiResponse<MuscleEntity>> call(String name) async {
    return await repository.getMuscleByName(name);
  }
}

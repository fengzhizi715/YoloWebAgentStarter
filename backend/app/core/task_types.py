from enum import StrEnum


class TaskType(StrEnum):
    DETECT = "detect"
    SEGMENT = "segment"


def annotation_type_for(task_type: TaskType | str) -> str:
    resolved = TaskType(task_type)
    return "bbox" if resolved is TaskType.DETECT else "polygon"


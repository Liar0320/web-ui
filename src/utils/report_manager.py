import os
import json
import uuid
import logging
import datetime
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class TaskExecutionRecord:
    """单次任务执行记录"""
    
    def __init__(self, task_description: str):
        self.task_id = str(uuid.uuid4())
        self.task_description = task_description
        self.start_time = datetime.datetime.now()
        self.end_time = None
        self.duration = None
        self.urls = []  # 访问的URL列表
        self.success = False
        self.error_message = None
    
    def add_url(self, url: str):
        """记录访问的URL"""
        if url and url not in self.urls:
            self.urls.append(url)
    
    def end_task(self, success: bool, error_message: Optional[str] = None):
        """结束任务记录"""
        self.end_time = datetime.datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.success = success
        self.error_message = error_message
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "duration_formatted": self.format_duration() if self.duration else None,
            "urls": self.urls,
            "success": self.success,
            "error_message": self.error_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskExecutionRecord':
        """从字典创建实例"""
        record = cls(data["task_description"])
        record.task_id = data["task_id"]
        record.start_time = datetime.datetime.fromisoformat(data["start_time"])
        if data.get("end_time"):
            record.end_time = datetime.datetime.fromisoformat(data["end_time"])
        record.duration = data.get("duration")
        record.urls = data.get("urls", [])
        record.success = data.get("success", False)
        record.error_message = data.get("error_message")
        return record
    
    def format_duration(self) -> str:
        """格式化持续时间"""
        if not self.duration:
            return "未完成"
        
        minutes, seconds = divmod(self.duration, 60)
        hours, minutes = divmod(minutes, 60)
        
        if hours > 0:
            return f"{int(hours)}小时{int(minutes)}分钟{int(seconds)}秒"
        elif minutes > 0:
            return f"{int(minutes)}分钟{int(seconds)}秒"
        else:
            return f"{int(seconds)}秒"


class ReportManager:
    """任务报表管理器"""
    
    def __init__(self, history_file: str = "./tmp/reports/history.json"):
        self.history_file = history_file
        self.current_task: Optional[TaskExecutionRecord] = None
        self.task_history: List[TaskExecutionRecord] = []
        self.load_history()
    
    def start_task_record(self, task_description: str) -> TaskExecutionRecord:
        """开始记录新任务"""
        self.current_task = TaskExecutionRecord(task_description)
        return self.current_task
    
    def record_url(self, url: str):
        """记录URL访问"""
        if self.current_task:
            self.current_task.add_url(url)
    
    def end_task_record(self, success: bool, error_message: Optional[str] = None) -> Optional[TaskExecutionRecord]:
        """结束当前任务记录并保存"""
        if not self.current_task:
            logger.warning("尝试结束不存在的任务记录")
            return None
        
        self.current_task.end_task(success, error_message)
        self.task_history.append(self.current_task)
        self.save_history()
        
        completed_task = self.current_task
        self.current_task = None
        return completed_task
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取累计统计信息"""
        total_executions = len(self.task_history)
        successful_executions = sum(1 for task in self.task_history if task.success)
        failed_executions = total_executions - successful_executions
        
        # 计算总运行时长（秒）
        total_duration = sum(task.duration or 0 for task in self.task_history)
        
        # 计算成功率
        success_rate = 0
        if total_executions > 0:
            success_rate = (successful_executions / total_executions) * 100
            
        # 计算平均运行时长
        avg_duration = 0
        if total_executions > 0:
            avg_duration = total_duration / total_executions
            
        # 计算成功任务的平均运行时长
        successful_tasks_duration = sum(task.duration or 0 for task in self.task_history if task.success)
        avg_successful_duration = 0
        if successful_executions > 0:
            avg_successful_duration = successful_tasks_duration / successful_executions
        
        # 格式化总运行时长
        minutes, seconds = divmod(total_duration, 60)
        hours, minutes = divmod(minutes, 60)
        total_duration_formatted = f"{int(hours)}小时{int(minutes)}分钟{int(seconds)}秒"
        
        # 格式化平均运行时长
        minutes, seconds = divmod(avg_duration, 60)
        hours, minutes = divmod(minutes, 60)
        avg_duration_formatted = f"{int(hours)}小时{int(minutes)}分钟{int(seconds)}秒"
        
        # 格式化成功任务平均运行时长
        minutes, seconds = divmod(avg_successful_duration, 60)
        hours, minutes = divmod(minutes, 60)
        avg_successful_duration_formatted = f"{int(hours)}小时{int(minutes)}分钟{int(seconds)}秒"
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": f"{success_rate:.2f}%",
            "total_duration": total_duration,
            "total_duration_formatted": total_duration_formatted,
            "avg_duration": avg_duration,
            "avg_duration_formatted": avg_duration_formatted,
            "avg_successful_duration": avg_successful_duration,
            "avg_successful_duration_formatted": avg_successful_duration_formatted
        }
    
    def load_history(self):
        """加载历史记录"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.task_history = [TaskExecutionRecord.from_dict(item) for item in data]
                logger.info(f"已加载 {len(self.task_history)} 条任务历史记录")
        except Exception as e:
            logger.error(f"加载历史记录失败: {e}")
            self.task_history = []
    
    def save_history(self):
        """保存历史记录"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                data = [task.to_dict() for task in self.task_history]
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"已保存 {len(self.task_history)} 条任务历史记录")
        except Exception as e:
            logger.error(f"保存历史记录失败: {e}")
    
    def export_excel(self, output_path: str = "./tmp/reports/task_report.xlsx") -> Optional[str]:
        """导出Excel报表"""
        try:
            # 检查pandas是否可用
            try:
                import pandas as pd
                logger.info("pandas 已成功导入")
            except ImportError as e:
                logger.error(f"导入pandas时出错: {e}")
                return None
                
            # 检查openpyxl是否可用
            try:
                import openpyxl
                logger.info("openpyxl 已成功导入，版本: {0}".format(openpyxl.__version__))
            except ImportError as e:
                logger.error(f"导入openpyxl时出错: {e}")
                return None
                
            # 确保目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 创建任务明细数据框
            task_records = []
            for task in self.task_history:
                record = {
                    "任务ID": task.task_id,
                    "任务描述": task.task_description,
                    "开始时间": task.start_time,
                    "结束时间": task.end_time,
                    "执行时长": task.format_duration(),
                    "访问页面链路": "\n".join(task.urls),
                    "是否成功": "成功" if task.success else "失败",
                    "错误信息": task.error_message or ""
                }
                task_records.append(record)
            
            df_tasks = pd.DataFrame(task_records)
            
            # 获取统计信息
            stats = self.get_statistics()
            
            # 创建统计信息数据框
            stats_data = [
                ["总执行次数", stats["total_executions"]],
                ["成功次数", stats["successful_executions"]],
                ["失败次数", stats["failed_executions"]],
                ["成功率", stats["success_rate"]],
                ["总运行时长", stats["total_duration_formatted"]],
                ["总任务平均每次耗时", stats["avg_duration_formatted"]],
                ["完成任务平均每次耗时", stats["avg_successful_duration_formatted"]]
            ]
            df_stats = pd.DataFrame(stats_data, columns=["统计项", "数值"])
            
            # 创建日期分组统计
            date_stats = {}
            for task in self.task_history:
                date_str = task.start_time.strftime('%Y-%m-%d')
                if date_str not in date_stats:
                    date_stats[date_str] = {"total": 0, "success": 0, "failed": 0, "duration": 0}
                
                date_stats[date_str]["total"] += 1
                if task.success:
                    date_stats[date_str]["success"] += 1
                else:
                    date_stats[date_str]["failed"] += 1
                date_stats[date_str]["duration"] += task.duration or 0
            
            date_records = []
            for date_str, data in date_stats.items():
                minutes, seconds = divmod(data["duration"], 60)
                hours, minutes = divmod(minutes, 60)
                duration_str = f"{int(hours)}小时{int(minutes)}分钟{int(seconds)}秒"
                
                record = {
                    "日期": date_str,
                    "执行次数": data["total"],
                    "成功次数": data["success"],
                    "失败次数": data["failed"],
                    "总运行时长": duration_str
                }
                date_records.append(record)
            
            df_dates = pd.DataFrame(date_records)
            
            # 尝试创建Excel文件
            try:
                # 创建Excel文件
                logger.info(f"尝试创建Excel文件: {output_path}")
                with pd.ExcelWriter(output_path) as writer:
                    logger.info("写入任务执行记录表")
                    df_tasks.to_excel(writer, sheet_name='任务执行记录', index=False)
                    logger.info("写入累计统计表")
                    df_stats.to_excel(writer, sheet_name='累计统计', index=False)
                    logger.info("写入日期统计表")
                    df_dates.to_excel(writer, sheet_name='日期统计', index=False)
                
                logger.info(f"报表已导出至: {output_path}")
                return output_path
            except Exception as e:
                logger.error(f"写入Excel文件时出错: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return None
        
        except Exception as e:
            logger.error(f"导出Excel报表失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

# 创建全局报表管理器实例
_global_report_manager = ReportManager()

def get_report_manager() -> ReportManager:
    """获取全局报表管理器实例"""
    global _global_report_manager
    return _global_report_manager 
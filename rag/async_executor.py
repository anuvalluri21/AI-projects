"""Fixed-size asynchronous worker pools."""
import asyncio

class AsyncExecutor:
    @staticmethod
    async def bounded_map(items, workers, operation):
        """Fixed workers avoid allocating one task per chunk for large PDFs."""
        iterator = iter(items)
        results = []

        async def worker():
            for item in iterator:
                results.append(await operation(item))

        async with asyncio.TaskGroup() as group:
            for _ in range(workers):
                group.create_task(worker())
        return results

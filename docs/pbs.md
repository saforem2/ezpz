# 🍋 `ezpz`: PBS Job Management

## 🤔 Determine Details of Currently Active Job

1. Find all currently running[^semantics] jobs owned by the user.
2. For each of these running jobs, build a dictionary of the form:

    ```python
    >>> jobs = ezpz.pbs.get_users_running_pbs_jobs()
    >>> jobs
    {
        jobid_A: [host_A0, host_A1, host_A2, ..., host_AN],
        jobid_B: [host_B0, host_B1, host_B2, ..., host_BN],
        ...,
    }
    ```

3. Look for _our_ `hostname` in the list of hosts for each job.
   - If found, we know we are participating in that job.

4. Once we have the `PBS_JOBID` of the job containing our `hostname`,
   we can find the `hostfile` for that job.
   - The `hostfile` is located in `/var/spool/pbs/aux/`.
   - The filename is of the form `jobid.hostname`.

5. ✅ Done!

   Example:

   ```python
   jobid = ezpz.pbs.get_pbs_jobid_of_active_job()
   num_nodes = len(jobs[jobid])
   world_size = num_nodes * ezpz.get_gpus_per_node()
   ```

[^semantics]: |
    - **Running**: Can have _multiple_ PBS jobs running at the same time
    - **Active**: Can only have _one_ active PBS job at a time
        - This is the job that we are **currently running on**

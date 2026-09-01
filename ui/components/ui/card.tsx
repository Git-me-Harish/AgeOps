import clsx from 'clsx';
import type { HTMLAttributes, ReactNode } from 'react';

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={clsx('card', className)} {...props} />;
}

export function CardHeader({ title, action }: { title: string; action?: ReactNode }) {
  return (
    <div className="card-header">
      <span className="card-title">{title}</span>
      {action}
    </div>
  );
}

export function CardBody({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={clsx('card-body', className)} {...props} />;
}
